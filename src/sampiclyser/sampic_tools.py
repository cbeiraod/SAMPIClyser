# -*- coding: utf-8 -*-
#############################################################################
# zlib License
#
# (C) 2025 Cristóvão Beirão da Cruz e Silva <cbeiraod@cern.ch>
#
# This software is provided 'as-is', without any express or implied
# warranty.  In no event will the authors be held liable for any damages
# arising from the use of this software.
#
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
#
# 1. The origin of this software must not be misrepresented; you must not
#    claim that you wrote the original software. If you use this software
#    in a product, an acknowledgment in the product documentation would be
#    appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
#    misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.
#############################################################################

import datetime
import math
import struct
from collections import Counter
from pathlib import Path
from typing import Iterator
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Union

import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

# import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import uproot
from matplotlib.dates import AutoDateFormatter
from matplotlib.dates import AutoDateLocator
from matplotlib.ticker import FormatStrFormatter
from pyarrow import RecordBatch
from scipy.signal import resample
from scipy.signal import resample_poly


def open_hit_reader(
    file_path: Path, cols: Sequence[str], batch_size: int = 100_000, root_tree: str = "sampic_hits"
) -> Iterator[Union[RecordBatch, ak.highlevel.Array]]:
    """
    Stream selected columns from a SAMPIC output file in memory-efficient batches.

    This function reads only the specified columns from large data files (Parquet,
    Feather, or ROOT) in fixed-size batches, yielding either Arrow RecordBatches
    (for Parquet/Feather) or dictionaries of NumPy arrays (for ROOT).

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the input file. Supported extensions:
        - `.parquet` or `.pq` for Parquet
        - `.feather` for Arrow Feather IPC
        - `.root` for ROOT files containing a TTree named `root_tree`
    cols : sequence of str
        Names of the columns (or ROOT branches) to read.
    batch_size : int, optional
        Maximum number of rows/entries per yielded batch (default: 100000).
    root_tree : str, optional
        Name of the ROOT TTree to read from (default: "sampic_hits").

    Yields
    ------
    RecordBatch or dict
        - For Parquet/Feather: `pyarrow.RecordBatch` containing the requested columns.
        - For ROOT: dict mapping branch names to NumPy arrays for each batch.

    Raises
    ------
    ValueError
        If the file extension is not among the supported types.
    """
    suffix = file_path.suffix.lower()
    if suffix in ('.parquet', '.pq'):
        pqf = pq.ParquetFile(str(file_path))
        yield from pqf.iter_batches(batch_size=batch_size, columns=list(cols))

    elif suffix == ".feather":
        dataset = ds.dataset(str(file_path), format="feather")
        scanner = dataset.scanner(batch_size=batch_size, columns=list(cols))
        yield from scanner.to_batches()

    elif suffix == ".root":
        tree_path = f"{file_path}:{root_tree}"
        yield from uproot.iterate(tree_path, list(cols), step_size=batch_size)

    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def get_channel_hits(file_path: Path, batch_size: int = 100_000, root_tree: str = "sampic_hits") -> pd.DataFrame:
    """
    Compute per-channel hit counts by streaming only the 'Channel' column.

    Supports Feather, Parquet, or ROOT (.root) files written by the Sampic decoder.
    Reads data in batches (to bound memory use) and tallies the number of rows
    (hits) observed on each channel.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the input data file.  Must have suffix `.feather`, `.parquet`, or `.root`.
    batch_size : int, optional
        Number of entries to read per iteration (default: 100000).
    root_tree : str, optional
        Name of the TTree inside the ROOT file to read (only used if `file_path` is `.root`;
        default: `"sampic_hits"`).

    Returns
    -------
    pandas.DataFrame
        A DataFrame with two columns:

        - `Channel` (int): channel identifier
        - `Hits`    (int): total number of hits on that channel

        Rows are sorted by increasing `Channel`.

    Raises
    ------
    ValueError
        If the file suffix is not one of `.feather`, `.parquet`, or `.root`.
    """
    counts = Counter()

    for batch in open_hit_reader(file_path=file_path, cols=["Channel"], batch_size=batch_size, root_tree=root_tree):
        arr = batch["Channel"].to_numpy()

        uniques, cnts = np.unique(arr, return_counts=True)
        for ch, cnt in zip(uniques, cnts):
            counts[int(ch)] += int(cnt)

    # Build and return the summary DataFrame
    df = pd.DataFrame(sorted(counts.items()), columns=["Channel", "Hits"])
    return df


def plot_channel_hits(
    df: pd.DataFrame,
    first_channel: int,
    last_channel: int,
    cms_label: str = "PPS",
    log_y: bool = False,
    figsize: tuple[float, float] = (6, 4),
    rlabel: str = "(13 TeV)",
    is_data: bool = True,
    color="C0",
    title: str | None = None,
) -> plt.Figure:
    """
    Draw a CMS-style bar histogram of hit counts per channel.

    Parameters
    ----------
    df : pandas.DataFrame
        Summary table with two columns:
        - `Channel` (int): channel indices
        - `Hits`    (int): hit counts per channel
    first_channel : int
        Lowest channel index to include on the x-axis.
    last_channel : int
        Highest channel index to include on the x-axis.
    cms_label : str, optional
        Text label for the CMS experiment (default: "PPS").
    log_y : bool, optional
        If True, use a logarithmic y-axis (default: False).
    figsize : tuple of float, optional
        Figure size in inches as (width, height) (default: (6, 4)).
    rlabel : str, optional
        Right-hand text label, typically collision energy (default: "(13 TeV)").
    is_data : bool, optional
        If True, annotate the plot as “Data”; if False, annotate as “Simulation”
        (default: True).
    color : any, optional
        Matplotlib color spec for the bars (default: "C0").
    title : str or None, optional
        Main title displayed above the axes; if None, no title is shown.

    Returns
    -------
    matplotlib.figure.Figure
        The Figure object containing the histogram.

    Raises
    ------
    ValueError
        If `last_channel` is less than `first_channel`.

    Notes
    -----
    - Channels missing from `df` are shown with zero hits.
    - In linear mode, y-axis tick labels are formatted in uppercase scientific
      notation (e.g. "4.0E6").
    - The plot uses `mplhep.style.CMS` with `cms_label` and `rlabel` positioned
      according to CMS styling conventions.
    - The `is_data` flag controls the “Data” vs. “Simulation” annotation.
    """
    # Build the full channel range and corresponding hit counts (0 if missing)
    channels = list(range(first_channel, last_channel + 1))
    hits_map = dict(zip(df["Channel"], df["Hits"]))
    counts = [hits_map.get(ch, 0) for ch in channels]

    # Apply CMS style from mplhep
    plt.style.use(hep.style.CMS)

    # Create figure and axis with custom size and create the bar histogram
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(channels, counts, align='center', width=1.0, edgecolor='black', color=color)

    # CMS label with customizable right text
    hep.cms.label(cms_label, data=is_data, rlabel=rlabel, loc=0, ax=ax)

    # Optional main title
    if title:
        ax.set_title(title, pad=12, weight="bold")

    # Y-axis scale and formatting
    if log_y:
        ax.set_yscale('log')
    else:
        # scientific notation for linear scale
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1E'))

    # Axis labels and limits
    ax.set_xlabel("Channel")
    ax.set_ylabel("Hits per Channel")
    ax.set_xlim(first_channel - 0.5, last_channel + 0.5)
    ax.set_xticks(channels)

    plt.tight_layout()
    # plt.show()

    return fig


def decode_byte_metadata(byte_metadata: dict[bytes, bytes]) -> dict[str, object]:
    """
    Decode raw byte-to-byte metadata into native Python types.

    Parameters
    ----------
    byte_metadata : dict of bytes → bytes
        Mapping of raw metadata keys and values as read from an Arrow or Parquet file.
        Keys and values are both byte strings.

    Returns
    -------
    metadata : dict of str → object
        Decoded metadata where each key is ASCII-decoded, and each value is converted
        according to its semantic type:

        - **str**:
          - Version/info fields such as ``software_version``,
            ``sampic_mezzanine_board_version``, ``ctrl_fpga_firmware_version``,
            ``sampling_frequency``, ``hit_number_format``, etc.
        - **datetime.datetime**:
          - The ``timestamp`` field, unpacked from a little-endian float64.
        - **int**:
          - Numeric fields such as ``num_channels`` and ``enabled_channels_mask``,
            unpacked from little-endian uint32.
        - **bool**:
          - Flag fields such as ``reduced_data_type``, ``without_waveform``,
            ``tdc_like_files``, ``inl_correction``, and ``adc_correction``,
            where a single zero byte means False and any other byte means True.

    Raises
    ------
    KeyError
        If a required metadata key is missing from the input dictionary.
    struct.error
        If unpacking a numeric or timestamp field fails due to incorrect byte length.

    Notes
    -----
    - Entries whose keys decode to ``'ARROW:schema'`` or ``'pandas'`` are ignored.
    - Any unrecognized keys will still be included in the output as their raw ASCII-decoded
      byte sequence, with the byte value left unchanged.
    """
    metadata: dict[str, object] = {}

    for key_bytes, data_bytes in byte_metadata.items():
        key = key_bytes.decode('ascii')
        # Skip Arrow internal metadata
        if key in ['ARROW:schema', 'pandas']:
            continue
        # Default: store raw bytes, will be overwritten if matched below
        value: object = data_bytes

        # Text fields
        if key in [
            'software_version',
            'sampic_mezzanine_board_version',
            'ctrl_fpga_firmware_version',
            'sampling_frequency',
            'hit_number_format',
            'unix_time_format',
            'data_format',
            'trigger_position_format',
            'data_samples_format',
        ]:
            value = data_bytes.decode('ascii')
        # Timestamp: little-endian 8-byte float
        elif key == 'timestamp':
            (ts,) = struct.unpack('<d', data_bytes)
            value = datetime.datetime.fromtimestamp(ts)
        # Unsigned int fields
        elif key in ['num_channels', 'enabled_channels_mask']:
            (tmp,) = struct.unpack('<I', data_bytes)
            value = tmp
        # Boolean flags: 0x00 => False, else True
        elif key in ['reduced_data_type', 'without_waveform', 'tdc_like_files', 'inl_correction', 'adc_correction']:
            value = False if data_bytes == b'\x00' else True

        metadata[key] = value
    return metadata


def load_root_metadata(file_path: str) -> dict[str, object]:
    """
    Read metadata from a 'metadata' TTree in a ROOT file and decode to Python types.

    Parameters
    ----------
    file_path : str
        Filesystem path to the ROOT file containing a TTree named 'metadata' with
        two branches: 'key' and 'value'.  Both branches should contain strings.

    Returns
    -------
    metadata : dict of str → object
        Dictionary mapping each metadata key to a Python value, converted as follows:

        - **datetime.datetime**
          If the key is `'timestamp'`, the string is parsed via
          `datetime.datetime.fromisoformat`.
        - **int**
          For `'num_channels'` and `'enabled_channels_mask'`, the string is cast to `int`.
        - **bool**
          For flags (`'reduced_data_type'`, `'without_waveform'`,
          `'tdc_like_files'`, `'inl_correction'`, `'adc_correction'`),
          the string `'False'` → `False`, all other values → `True`.
        - **str**
          All other entries are left as Python strings.

    Raises
    ------
    KeyError
        If the TTree 'metadata' or the branches 'key'/'value' are not found.
    ValueError
        If a timestamp string cannot be parsed by `fromisoformat`, or if an
        integer conversion fails.

    Notes
    -----
    - This function uses `uproot.open` to read the ROOT file in read-only mode.
    - It expects the metadata tree to have exactly two branches, `'key'` and
      `'value'`, both containing arrays of equal length.
    """
    metadata: dict[str, object] = {}
    with uproot.open(file_path) as f:
        # Expect a TTree named 'metadata' with branches 'key' and 'value'
        arr = f['metadata'].arrays(['key', 'value'], library='np')
        for key_bytes, val_arr in zip(arr['key'], arr['value']):
            key = key_bytes.decode('ascii') if isinstance(key_bytes, (bytes, bytearray)) else str(key_bytes)
            raw = val_arr
            # Parse types
            if key == 'timestamp':
                value = datetime.datetime.fromisoformat(raw)
            elif key in ['num_channels', 'enabled_channels_mask']:
                value = int(raw)
            elif key in ['reduced_data_type', 'without_waveform', 'tdc_like_files', 'inl_correction', 'adc_correction']:
                value = False if raw == 'False' else True
            else:
                # Default: raw may be bytes or numpy scalar
                if isinstance(raw, bytes):
                    value = raw.decode('ascii')
                else:
                    value = str(raw)
            metadata[key] = value
    return metadata


def get_file_metadata(file_path: Path) -> dict[str, object]:
    """
    Load metadata from a SAMPIC output file, selecting the appropriate reader.

    This function examines the file extension of `file_path` and invokes the
    corresponding metadata decoder:

    - **Parquet** (`.parquet`, `.pq`): uses `pyarrow.parquet` metadata and
      `decode_byte_metadata` for byte-to-type conversion.
    - **Feather** (`.feather`): uses `pyarrow.ipc` schema metadata and
      `decode_byte_metadata`.
    - **ROOT** (`.root`): uses `uproot` to read a ´metadata´ TTree via
      `load_root_metadata`.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the input file whose metadata to extract.  Supported suffixes
        are `.parquet`, `.pq`, `.feather`, and `.root`.

    Returns
    -------
    metadata : dict of str → object
        Dictionary of metadata fields mapped to native Python values, where
        each value may be one of:

        - **str**
          For textual fields (software versions, format strings).
        - **int**
          For numeric fields (e.g. `num_channels`, masks).
        - **bool**
          For flag fields (`reduced_data_type`, etc.).
        - **datetime.datetime**
          For timestamp fields.

    Raises
    ------
    ValueError
        If `file_path` has an unsupported suffix or if metadata loading fails
        for any reason.
    """
    suffix = file_path.suffix.lower()
    if suffix in ('.parquet', '.pq'):
        pqf = pq.ParquetFile(str(file_path))
        return decode_byte_metadata(pqf.metadata.metadata or {})
    elif suffix == '.feather':
        ipcf = pa.ipc.open_file(str(file_path))
        return decode_byte_metadata(ipcf.schema.metadata or {})
    elif suffix == '.root':
        return load_root_metadata(str(file_path))
    else:
        raise ValueError(f"Unsupported format: {file_path.suffix}")


def get_period_from_file_metadata(metadata: dict[str, object]) -> float:
    """
    Compute the sampling period (seconds per sample) from file metadata.

    Parameters
    ----------
    metadata : dict of str → object
        Dictionary of file metadata. Must contain the key
        ``'sampling_frequency'`` whose value is a string of the form
        ``"<freq> <unit>"``, where:

        - ``<freq>`` is a floating-point number (e.g. "5.0")
        - ``<unit>`` is either `"MS/s"` (megasamples per second) or
          `"kS/s"` (kilosamples per second).

    Returns
    -------
    period : float
        Time interval between consecutive samples, in seconds.

    Raises
    ------
    KeyError
        If the metadata dict does not include the `"sampling_frequency"` key.
    RuntimeError
        If the unit parsed from `"sampling_frequency"` is not one of
        `"MS/s"` or `"kS/s"`.

    Examples
    --------
    >>> meta = {"sampling_frequency": "5 MS/s"}
    >>> get_period_from_file_metadata(meta)
    2e-07
    >>> meta = {"sampling_frequency": "10 kS/s"}
    >>> get_period_from_file_metadata(meta)
    0.0001
    """
    freq, rate = metadata['sampling_frequency'].split(' ')
    freq = float(freq)

    if rate == "MS/s":
        period = 1.0 / (freq * 1e6)
    elif rate == "kS/s":
        period = 1.0 / (freq * 1e3)
    else:
        raise RuntimeError(f"Unknown rate: {rate}")

    return period


def plot_hit_rate(  # noqa: max-complexity=22
    file_path: Path,
    bin_size: float = 1.0,
    batch_size: int = 100_000,
    plot_hits: bool = False,
    start_time: datetime.datetime | float | None = None,
    end_time: datetime.datetime | float | None = None,
    root_tree: str = "sampic_hits",
    scale_factor: float = 1.0,
    cms_label: str = "PPS",
    log_y: bool = False,
    figsize: tuple[float, float] = (6, 4),
    rlabel: str = "(13 TeV)",
    is_data: bool = True,
    color="C0",
    title: str | None = None,
) -> plt.Figure:
    """
    Plot the hit rate (or raw hits) as a function of time from large data files.

    Streams the “UnixTime” column in batches from a Feather, Parquet, or ROOT file,
    bins events into fixed-width time intervals, and renders a CMS-style time series.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the input data file; supported suffixes are `.feather`, `.parquet`, `.pq`, and `.root`.
    bin_size : float, optional
        Width of each time bin in seconds; values below 0.1 are rounded up to 0.1 (default: 1.0).
    batch_size : int, optional
        Number of entries to read per I/O batch (default: 100000).
    plot_hits : bool, optional
        If True, plot the raw count per bin; otherwise plot the rate
        (count divided by `bin_size`) (default: False).
    start_time : datetime.datetime, float, or None, optional
        Start of the time window for plotting, as a datetime or UNIX timestamp.
        If None, uses the file's “start_of_run” metadata.  Aligned to the
        nearest lower multiple of `bin_size` (default: None).
    end_time : datetime.datetime, float, or None, optional
        End of the time window for plotting, as a datetime or UNIX timestamp.
        If None, determined from the data.  Aligned to the nearest upper
        multiple of `bin_size` (default: None).
    root_tree : str, optional
        Name of the TTree in a ROOT file (only used if `file_path` ends in `.root`;
        default: `"sampic_hits"`).
    scale_factor : float, optional
        Multiplier applied to each bin's count (e.g. to account for
        central trigger multiplicity) before plotting (default: 1.0).
    cms_label : str, optional
        CMS experiment label (default: `"PPS"`).
    log_y : bool, optional
        If True, use a logarithmic y-axis (default: False).
    figsize : tuple of float, optional
        Figure size in inches as (width, height) (default: (6, 4)).
    rlabel : str, optional
        Additional right-hand label (e.g. collision energy) (default: `"(13 TeV)"`).
    is_data : bool, optional
        If True, annotate plots as “Data”; if False, annotate as “Simulation”
        (default: True).
    color : color spec, optional
        Matplotlib color for the line or bars (default: `"C0"`).
    title : str or None, optional
        Main title for the figure; if None, no title is drawn (default: None).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the hit-rate (or hit-count) vs. time plot,
        styled according to CMS conventions.

    Raises
    ------
    ValueError
        If `file_path` has an unsupported suffix.

    Notes
    -----
    - Time bins are computed as `floor((t - t0)/bin_size)` indices,
      then shifted back to absolute times for plotting.
    - X-axis tick formatting uses Matplotlib's `AutoDateLocator` and
      `AutoDateFormatter` for sensible date/time labels across variable spans.
    """
    # enforce minimum bin size
    bin_size = max(bin_size, 0.1)

    # fetch run‐start from metadata; override if start_time provided
    metadata = get_file_metadata(file_path)
    run_start = metadata.get("timestamp")
    if isinstance(run_start, datetime.datetime):
        run_start_ts = run_start.timestamp()
    else:
        run_start_ts = float(run_start)
    # align to bin boundary
    run_start_ts = math.floor(run_start_ts / bin_size) * bin_size

    # apply user override
    if start_time is not None:
        st = start_time.timestamp() if isinstance(start_time, datetime.datetime) else float(start_time)
        if st > run_start_ts:
            run_start_ts = math.floor(st / bin_size) * bin_size

    counts = Counter()

    for batch in open_hit_reader(file_path=file_path, cols=["UnixTime"], batch_size=batch_size, root_tree=root_tree):
        arr = batch["UnixTime"].to_numpy()

        for t in arr:
            idx = int((t - run_start_ts) // bin_size)
            if idx >= 0:
                counts[idx] += 1

    if not counts:
        raise RuntimeError("No hits found in file.")

    for idx in range(max(counts.keys())):
        if idx not in counts:
            counts[idx] = 0

    # Build sorted time and rate arrays
    bins = np.array(sorted(counts.keys()), dtype=int)
    times = bins * bin_size + run_start_ts

    # apply end_time override
    if end_time is not None:
        et = end_time.timestamp() if isinstance(end_time, datetime.datetime) else float(end_time)
        max_bin = math.ceil((et - run_start_ts) / bin_size)
        mask = bins <= max_bin
        bins = bins[mask]
        times = bins * bin_size + run_start_ts

    # convert to datetime for plotting
    dtimes = [datetime.datetime.fromtimestamp(ts) for ts in times]
    if plot_hits:
        rates = np.array([counts[b] * scale_factor for b in bins], dtype=int)
    else:
        rates = np.array([counts[b] * scale_factor / bin_size for b in bins], dtype=int)

    # Plot
    plt.style.use(hep.style.CMS)

    # Create figure and axis with custom size and create the plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.step(dtimes, rates, where="mid", color=color)

    # CMS label with customizable right text
    hep.cms.label(cms_label, data=is_data, rlabel=rlabel, loc=0, ax=ax)

    # Optional main title
    if title:
        ax.set_title(title, pad=12, weight="bold")

    # Y-axis scale and formatting
    if log_y:
        ax.set_yscale('log')

    ax.set_xlabel("Time")
    if plot_hits:
        ax.set_ylabel(f"Hits per {bin_size:.1f} s")
    else:
        ax.set_ylabel("Hit Rate [Hz]")

    # date formatting
    locator = AutoDateLocator()
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.set_xlim(dtimes[0], dtimes[-1])

    # format x-axis as dates
    fig.autofmt_xdate()

    plt.tight_layout()

    return fig


def plot_channel_hit_rate(  # noqa: max-complexity=22
    file_path: Path,
    channel: int = 0,
    bin_size: float = 1.0,
    batch_size: int = 100_000,
    plot_hits: bool = False,
    start_time: datetime.datetime | float | None = None,
    end_time: datetime.datetime | float | None = None,
    root_tree: str = "sampic_hits",
    scale_factor: float = 1.0,
    cms_label: str = "PPS",
    log_y: bool = False,
    figsize: tuple[float, float] = (6, 4),
    rlabel: str = "(13 TeV)",
    is_data: bool = True,
    color="C0",
    title: str | None = None,
) -> plt.Figure:
    """
    Plot the hit rate (or raw hits) as a function of time from large data files.

    Streams the “UnixTime” column in batches from a Feather, Parquet, or ROOT file,
    bins events into fixed-width time intervals, and renders a CMS-style time series.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the input data file; supported suffixes are `.feather`, `.parquet`, `.pq`, and `.root`.
    channel : int, optional
        The SAMPIC channel to plot (default: 0).
    bin_size : float, optional
        Width of each time bin in seconds; values below 0.1 are rounded up to 0.1 (default: 1.0).
    batch_size : int, optional
        Number of entries to read per I/O batch (default: 100000).
    plot_hits : bool, optional
        If True, plot the raw count per bin; otherwise plot the rate
        (count divided by `bin_size`) (default: False).
    start_time : datetime.datetime, float, or None, optional
        Start of the time window for plotting, as a datetime or UNIX timestamp.
        If None, uses the file's “start_of_run” metadata.  Aligned to the
        nearest lower multiple of `bin_size` (default: None).
    end_time : datetime.datetime, float, or None, optional
        End of the time window for plotting, as a datetime or UNIX timestamp.
        If None, determined from the data.  Aligned to the nearest upper
        multiple of `bin_size` (default: None).
    root_tree : str, optional
        Name of the TTree in a ROOT file (only used if `file_path` ends in `.root`;
        default: `"sampic_hits"`).
    scale_factor : float, optional
        Multiplier applied to each bin's count (e.g. to account for
        central trigger multiplicity) before plotting (default: 1.0).
    cms_label : str, optional
        CMS experiment label (default: `"PPS"`).
    log_y : bool, optional
        If True, use a logarithmic y-axis (default: False).
    figsize : tuple of float, optional
        Figure size in inches as (width, height) (default: (6, 4)).
    rlabel : str, optional
        Additional right-hand label (e.g. collision energy) (default: `"(13 TeV)"`).
    is_data : bool, optional
        If True, annotate plots as “Data”; if False, annotate as “Simulation”
        (default: True).
    color : color spec, optional
        Matplotlib color for the line or bars (default: `"C0"`).
    title : str or None, optional
        Main title for the figure; if None, no title is drawn (default: None).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the hit-rate (or hit-count) vs. time plot,
        styled according to CMS conventions.

    Raises
    ------
    ValueError
        If `file_path` has an unsupported suffix.

    Notes
    -----
    - Time bins are computed as `floor((t - t0)/bin_size)` indices,
      then shifted back to absolute times for plotting.
    - X-axis tick formatting uses Matplotlib's `AutoDateLocator` and
      `AutoDateFormatter` for sensible date/time labels across variable spans.
    """
    # enforce minimum bin size
    bin_size = max(bin_size, 0.1)

    # fetch run‐start from metadata; override if start_time provided
    metadata = get_file_metadata(file_path)
    run_start = metadata.get("timestamp")
    if isinstance(run_start, datetime.datetime):
        run_start_ts = run_start.timestamp()
    else:
        run_start_ts = float(run_start)
    # align to bin boundary
    run_start_ts = math.floor(run_start_ts / bin_size) * bin_size

    # apply user override
    if start_time is not None:
        st = start_time.timestamp() if isinstance(start_time, datetime.datetime) else float(start_time)
        if st > run_start_ts:
            run_start_ts = math.floor(st / bin_size) * bin_size

    counts = Counter()

    for batch in open_hit_reader(file_path=file_path, cols=["Channel", "UnixTime"], batch_size=batch_size, root_tree=root_tree):
        # Duck‐type: try Arrow first, else assume Awkward
        # if hasattr(batch, "column"):
        #     # PyArrow RecordBatch
        #     ch_arr = batch.column("Channel").to_numpy()
        #     time_arr = batch.column("UnixTime").to_numpy()
        # else:
        #     # Awkward Array from uproot.iterate
        #     # convert to numpy via __array__ interface
        #     ch_arr = np.asarray(batch["Channel"])
        #     time_arr = np.asarray(batch["UnixTime"])
        ch_arr = batch["Channel"].to_numpy()
        time_arr = batch["UnixTime"].to_numpy()

        arr = time_arr[ch_arr == channel]
        for t in arr:
            idx = int((t - run_start_ts) // bin_size)
            if idx >= 0:
                counts[idx] += 1

    if not counts:
        raise RuntimeError("No hits found in file.")

    for idx in range(max(counts.keys())):
        if idx not in counts:
            counts[idx] = 0

    # Build sorted time and rate arrays
    bins = np.array(sorted(counts.keys()), dtype=int)
    times = bins * bin_size + run_start_ts

    # apply end_time override
    if end_time is not None:
        et = end_time.timestamp() if isinstance(end_time, datetime.datetime) else float(end_time)
        max_bin = math.ceil((et - run_start_ts) / bin_size)
        mask = bins <= max_bin
        bins = bins[mask]
        times = bins * bin_size + run_start_ts

    # convert to datetime for plotting
    dtimes = [datetime.datetime.fromtimestamp(ts) for ts in times]
    if plot_hits:
        rates = np.array([counts[b] * scale_factor for b in bins], dtype=int)
    else:
        rates = np.array([counts[b] * scale_factor / bin_size for b in bins], dtype=int)

    # Plot
    plt.style.use(hep.style.CMS)

    # Create figure and axis with custom size and create the plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.step(dtimes, rates, where="mid", color=color)

    # CMS label with customizable right text
    hep.cms.label(cms_label, data=is_data, rlabel=rlabel, loc=0, ax=ax)

    # Optional main title
    if title:
        ax.set_title(title, pad=12, weight="bold")

    # Y-axis scale and formatting
    if log_y:
        ax.set_yscale('log')

    ax.set_xlabel("Time")
    if plot_hits:
        ax.set_ylabel(f"Channel {channel} hits per {bin_size:.1f} s")
    else:
        ax.set_ylabel(f"Channel {channel} hit Rate [Hz]")

    # date formatting
    locator = AutoDateLocator()
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.set_xlim(dtimes[0], dtimes[-1])

    # format x-axis as dates
    fig.autofmt_xdate()

    plt.tight_layout()

    return fig


def windowed_sinc_interpolation(t_orig: np.ndarray, y_orig: np.ndarray, t_new: np.ndarray, window: str = 'hann', M: int = 8) -> np.ndarray:
    r"""
    Band-limited interpolation of a uniformly sampled signal using a windowed sinc kernel.

    Constructs a truncated sinc filter of half-width `M` samples, applies a smooth
    tapering window (Hann or Hamming) to reduce ringing, and convolves it with the
    input data to estimate values at new time points.

    Parameters
    ----------
    t_orig : ndarray of float, shape (N,)
        Original sample times (must be uniformly spaced).
    y_orig : ndarray of float, shape (N,)
        Original sample values.
    t_new : ndarray of float, shape (M_new,)
        Desired output times (must lie within the range of `t_orig`).
    window : {'hann', 'hamming'}, optional
        Type of tapering window to apply to the sinc kernel.  Default is 'hann'.
    M : int, optional
        Half-width of the truncated sinc kernel, in number of original samples.
        Total kernel length will be `2*M + 1`.  Default is 8.

    Returns
    -------
    y_new : ndarray of float, shape (M_new,)
        Interpolated values at `t_new`.

    Raises
    ------
    ValueError
        If `t_orig` is not at least two points, if `window` is not recognized,
        or if `t_new` contains values outside the range of `t_orig`.

    Notes
    -----
    - This implementation assumes `t_orig` is **uniformly** spaced.  If that is
      not the case, consider first resampling to a uniform grid or using a
      more general interpolator.
    - The Hann window is defined as
      $$ w[k] = 0.5 + 0.5\cos\bigl(2\pi k/(2M+1)\bigr),\quad k=-M\ldots M. $$
    - The Hamming window uses
      $$ w[k] = 0.54 + 0.46\cos\bigl(2\pi k/(2M+1)\bigr). $$
    - Each output sample `y_new[i]` is
      $$ \sum_{k=-M}^{M} y_{\!n}\,\text{sinc}\bigl((t_{\!new}-t_{\!orig,n})/T\bigr)\,w[k], $$
      where \(T\) is the uniform sample spacing and \(n\) is chosen so that
      \(t_{\!orig,n}\) is the nearest original sample to each \(t_{\!new,i}\).

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 1, 11)
    >>> y = np.sin(2*np.pi*5*t)
    >>> t_fine = np.linspace(0, 1, 101)
    >>> y_fine = windowed_sinc_interpolation(t, y, t_fine, window='hann', M=4)
    """
    # Validate inputs
    t_orig = np.asarray(t_orig, dtype=float)
    y_orig = np.asarray(y_orig, dtype=float)
    t_new = np.asarray(t_new, dtype=float)

    # Basic validation
    if t_orig.ndim != 1 or y_orig.ndim != 1:
        raise ValueError("t_orig and y_orig must be 1D arrays")
    if t_orig.size < 2:
        raise ValueError("Need at least two original samples for interpolation")
    if t_new.ndim != 1:
        raise ValueError("t_new must be a 1D array")
    if t_new.min() < t_orig.min() or t_new.max() > t_orig.max():
        raise ValueError("t_new values must lie within the range of t_orig")
    if window not in ('hann', 'hamming'):
        raise ValueError(f"Unknown window type '{window}'; use 'hann' or 'hamming'")

    # Uniform spacing
    T = t_orig[1] - t_orig[0]
    if not np.allclose(np.diff(t_orig), T, atol=1e-8 * T):
        raise ValueError("t_orig must be uniformly spaced")

    # Precompute windowed‐sinc kernel indices and window
    k = np.arange(-M, M + 1)
    if window == 'hann':
        w = 0.5 + 0.5 * np.cos(2 * np.pi * k / (2 * M + 1))
    else:  # 'hamming'
        w = 0.54 + 0.46 * np.cos(2 * np.pi * k / (2 * M + 1))
    w = w / w[M]  # normalize to 1 at center

    y_new = np.empty_like(t_new, dtype=float)

    # For each target time, center the kernel at the nearest original sample
    for i, tn in enumerate(t_new):
        # find nearest index in t_orig
        n0 = int(np.round((tn - t_orig[0]) / T))
        idx = n0 + k

        # mask out-of-bounds indices
        valid = (idx >= 0) & (idx < t_orig.size)
        ti = t_orig[idx[valid]]
        yi = y_orig[idx[valid]]
        wi = w[valid]

        # compute sinc((tn - ti)/T)
        sinc_vals = np.sinc((tn - ti) / T)
        y_new[i] = np.dot(yi * sinc_vals, wi)

    return y_new


def lanczos_interpolation(t_orig: np.ndarray, y_orig: np.ndarray, t_new: np.ndarray, a: int = 3) -> np.ndarray:
    """
    Interpolate a uniformly sampled signal using the Lanczos kernel.

    The Lanczos filter uses a windowed sinc kernel of order `a`:
    L(x) = sinc(x) · sinc(x / a) for |x| ≤ a, zero otherwise. This
    yields near-ideal bandlimited interpolation with reduced ringing.

    Parameters
    ----------
    t_orig : ndarray of float, shape (N,)
        Original, uniformly spaced sample times.
    y_orig : ndarray of float, shape (N,)
        Original sample values.
    t_new : ndarray of float, shape (M,)
        Desired output times (must lie within the range of `t_orig`).
    a : int, optional
        Lanczos order (kernel half-width in samples). Common choices are
        2 or 3. Default is 3.

    Returns
    -------
    y_new : ndarray of float, shape (M,)
        Interpolated values at `t_new`.

    Raises
    ------
    ValueError
        If `t_orig` and `y_orig` are not 1D arrays of equal length,
        or if `t_new` lies outside the range `[t_orig.min(), t_orig.max()]`,
        or if `a` is not a positive integer.

    Notes
    -----
    - Assumes uniform spacing in `t_orig`.  If spacing varies, results
      will be invalid.
    - For each `t_new[i]`, the kernel covers indices `k` from
      `⌈(t_new[i]-t_orig[0])/T⌉ - a + 1` to `⌊(t_new[i]-t_orig[0])/T⌋ + a`,
      where `T = t_orig[1] - t_orig[0]`.
    - Out-of-bounds sample indices are clipped to the valid range.
    - The kernel is defined as:
      ```
      L_n = sinc((t_new-t_orig[n])/T) * sinc((t_new-t_orig[n])/(a*T))
      ```
      and `y_new[i] = Σ_n y_orig[n] · L_n`.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 1, 11)
    >>> y = np.sin(2*np.pi*5*t)
    >>> t_fine = np.linspace(0, 1, 101)
    >>> y_fine = lanczos_interpolation(t, y, t_fine, a=3)
    """
    # Validate inputs
    t_orig = np.asarray(t_orig, dtype=float)
    y_orig = np.asarray(y_orig, dtype=float)
    t_new = np.asarray(t_new, dtype=float)

    if t_orig.ndim != 1 or y_orig.ndim != 1:
        raise ValueError("t_orig and y_orig must be 1D arrays")
    if t_orig.shape != y_orig.shape:
        raise ValueError("t_orig and y_orig must have the same length")
    if t_new.ndim != 1:
        raise ValueError("t_new must be a 1D array")
    if a < 1 or not isinstance(a, int):
        raise ValueError("Lanczos order 'a' must be a positive integer")
    if t_new.min() < t_orig.min() or t_new.max() > t_orig.max():
        raise ValueError("t_new values must lie within the range of t_orig")

    t_min, _ = t_orig[0], t_orig[-1]

    # Uniform sample interval
    T = t_orig[1] - t_orig[0]
    if not np.allclose(np.diff(t_orig), T, atol=1e-8 * abs(T)):
        raise ValueError("t_orig must be uniformly spaced")

    y_new = np.empty_like(t_new, dtype=float)

    # Precompute sinc denominator factor
    for i, tn in enumerate(t_new):
        # position in sample units
        x = (tn - t_min) / T
        m = int(np.floor(x))
        # kernel support indices
        k = np.arange(m - a + 1, m + a + 1, dtype=int)
        # Keep only in-bounds sample indices
        mask = (k >= 0) & (k < len(t_orig))
        k_clipped = k[mask]
        t_k = t_orig[k_clipped]
        y_k = y_orig[k_clipped]

        # compute lanczos kernel: sinc + windowed sinc
        arg = (tn - t_k) / T
        lanczos_kernel = np.sinc(arg) * np.sinc(arg / a)

        # Renormalize so the truncated kernel sums to 1
        lanczos_kernel /= np.sum(lanczos_kernel)

        y_new[i] = np.dot(y_k, lanczos_kernel)

    return y_new


def apply_interpolation_method(
    x_orig: np.ndarray,
    y_orig: np.ndarray,
    period: float,
    interpolation_method: Optional[str] = "sinc",
    interpolation_factor: int = 4,
    interpolation_parameter: int = 8,
    offset: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate a uniformly sampled waveform using various methods.

    Parameters
    ----------
    x_orig : ndarray, shape (N,)
        Original sample times (must be monotonically increasing and
        uniformly spaced by `period`).
    y_orig : ndarray, shape (N,)
        Original sample values.
    period : float
        Time interval between consecutive samples in `x_orig`.
    interpolation_method : {'sinc', 'hann', 'hamming', 'lanczos', 'resample', 'resample_poly'}, optional
        Which method to use:
        - `'sinc'`         : ideal sinc interpolation (no window)
        - `'hann'`         : windowed-sinc with Hann window
        - `'hamming'`      : windowed-sinc with Hamming window
        - `'lanczos'`      : Lanczos kernel of order `interpolation_parameter`
        - `'resample'`     : FFT-based resample via `scipy.signal.resample`
        - `'resample_poly'`: polyphase FIR via `scipy.signal.resample_poly`
        Default is `'sinc'`.
    interpolation_factor : int, optional
        Upsampling factor: number of output points = `len(x_orig) * interpolation_factor`.
        Must be ≥ 1.  Default is 4.
    interpolation_parameter : int, optional
        Secondary parameter for certain methods:
        - For `'hann'`/`'hamming'`, this is the half-width `M` of the windowed-sinc.
        - For `'lanczos'`, this is the Lanczos order `a`.
        - For `'resample_poly'`, this is the FIR window's beta (in a Kaiser window).
        Ignored by `'sinc'` and `'resample'`.  Default is 8.
    offset : float or None, optional
        Baseline offset to subtract before interpolation, then add back afterward.
        If None, treated as zero.  Default is None.

    Returns
    -------
    x_fine : ndarray, shape (N * interpolation_factor,)
        Uniformly spaced output time axis from `x_orig[0]` to `x_orig[-1]`.
    y_fine : ndarray, same shape as `x_fine`
        Interpolated sample values.

    Raises
    ------
    ValueError
        - If `x_orig` and `y_orig` have mismatched lengths or are not 1-D.
        - If `interpolation_factor` < 1.
        - If `interpolation_method` is unrecognized.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 1, 11)
    >>> y = np.sin(2*np.pi*5*t)
    >>> t_fine, y_fine = apply_interpolation_method(t, y, t[1]-t[0], interpolation_method='sinc')
    """
    # Input validation
    x_orig = np.asarray(x_orig, dtype=float)
    y_orig = np.asarray(y_orig, dtype=float)

    if x_orig.ndim != 1 or y_orig.ndim != 1:
        raise ValueError("x_orig and y_orig must be 1D arrays")
    if x_orig.shape != y_orig.shape:
        raise ValueError("x_orig and y_orig must have the same length")
    if interpolation_factor < 1:
        raise ValueError("interpolation_factor must be >= 1")
    offset = 0.0 if offset is None else float(offset)

    # Build the fine grid once
    num_fine = len(x_orig) * interpolation_factor
    x_fine = np.linspace(x_orig[0], x_orig[-1], num_fine)

    # Dispatch to the chosen method
    method = interpolation_method.lower() if interpolation_method else ""
    if method == "sinc":
        kernel = np.sinc((x_fine[:, None] - x_orig[None, :]) / period)
        y_fine = kernel.dot(y_orig - offset) + offset

    elif method in ("hann", "hamming"):
        y_fine = (
            windowed_sinc_interpolation(t_orig=x_orig, y_orig=y_orig - offset, t_new=x_fine, window=method, M=interpolation_parameter)
            + offset
        )

    elif method == "lanczos":
        y_fine = lanczos_interpolation(t_orig=x_orig, y_orig=y_orig - offset, t_new=x_fine, a=interpolation_parameter) + offset

    elif method == "resample":
        # scipy.signal.resample returns (y_new, x_new) if given t=x_orig
        y_fine, x_temp = resample(y_orig - offset, num=num_fine, t=x_orig)
        # override x_fine to match returned times
        x_fine = x_temp
        y_fine = y_fine + offset

    elif method == "resample_poly":
        y_fine = (
            resample_poly(
                x=y_orig - offset, up=interpolation_factor, down=1, window=('kaiser', interpolation_parameter), padtype='constant', cval=0.0
            )
            + offset
        )

    else:
        raise ValueError(f"Unknown interpolation method: {interpolation_method!r}")

    return x_fine, y_fine


def select_waveforms(
    batches: Iterator[Union[RecordBatch, ak.highlevel.Array]], first_hit: int, num_hits: int, channel_filter: Optional[Set[int]] = None
) -> Iterator[Tuple[int, float, int, np.ndarray, np.ndarray]]:
    """
    Flatten record batches or Awkward arrays into individual waveform records,
    with optional hit-index slicing and channel filtering.

    Parameters
    ----------
    batches : iterator of RecordBatch or awkward.highlevel.Array
        Stream of data blocks each containing the fields
        `'HITNumber'`, `'Channel'`, `'Baseline'`, `'DataSize'`, `'TriggerPosition'`, and `'DataSample'`.
    first_hit : int
        Number of initial hits to skip before yielding.
    num_hits : int
        Maximum number of hits to yield after skipping `first_hit`.
    channel_filter : set of int or None, optional
        If provided, only waveforms whose channel index is in this set are yielded.

    Yields
    ------
    hit_number : int
        The sequential hit number given by SAMPIC to this waveform.
    channel : int
        SAMPIC channel index for this waveform.
    baseline : float
        Baseline offset for the waveform.
    n_samples : int
        Number of ADC samples in this waveform.
    trigger_positions : ndarray of int
        1D array of length `n_samples`, with 0/1 indicating trigger positions.
    samples : ndarray of float
        1D array of length `n_samples` containing the ADC values.

    Raises
    ------
    ValueError
        If a batch is missing any of the required fields.

    Notes
    -----
    - Uses duck typing to detect a PyArrow RecordBatch (has `.column()`) vs.
      an Awkward Array (indexable by field name).
    - Stops iteration once `num_hits` waveforms have been yielded.
    """
    required_fields = {"HITNumber", "Channel", "Baseline", "DataSize", "TriggerPosition", "DataSample"}
    count = 0
    yielded = 0

    for batch in batches:
        # Verify required fields exist
        batch_fields = set(batch.schema.names) if isinstance(batch, RecordBatch) else set(batch.fields)
        missing = required_fields - batch_fields
        if missing:
            raise ValueError(f"Batch is missing required fields: {missing}")

        # Extract common columns
        hitids = batch['HITNumber'].to_numpy()
        channels = batch['Channel'].to_numpy()
        baselines = batch['Baseline'].to_numpy()
        sizes = batch['DataSize'].to_numpy()

        # Extract trigger positions and samples, allowing copy for Arrow
        if hasattr(batch, "column"):
            triggers = batch["TriggerPosition"].to_numpy(zero_copy_only=False)
            samples = batch["DataSample"].to_numpy(zero_copy_only=False)
        else:
            triggers = batch["TriggerPosition"].to_numpy()
            samples = batch["DataSample"].to_numpy()

        # Iterate waveform by waveform
        for hid, ch, bl, n, tp, data in zip(hitids, channels, baselines, sizes, triggers, samples):
            # Channel filter
            if channel_filter is not None and ch not in channel_filter:
                count += 1
                continue

            # Skip until first_hit
            if count < first_hit:
                count += 1
                continue

            # Stop after num_hits
            if yielded >= num_hits:
                return

            # Yield the waveform tuple
            yield int(hid), int(ch), float(bl), int(n), np.asarray(tp), np.asarray(data)
            count += 1
            yielded += 1


def reorder_circular_samples_with_trigger(
    trig_arr: np.ndarray,
    samp_arr: np.ndarray,
    reorder_samples: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate a circular buffer so that the contiguous trigger block (1s) appears at the end.

    Verify that all trigger markers (1s) in a circular array are contiguous,
    then rotate both the trigger array and optionally the associated sample array so
    that the block of 1s appears at the end of the array.

    This is useful when interpreting a circular buffer of ADC samples where
    the trigger position-marked by one or more consecutive 1s—may wrap
    around the end of the buffer.  After reordering, the data will be in true
    time-order, with the trigger block at the end.

    In circular context, a block of 1s may wrap from the end back to the start
    of the array (e.g. [1,0,0,1] is a valid 2-wide trigger block).  This function
    verifies that all 1s form exactly one circularly-contiguous block, then
    performs a roll so that those 1s occupy the last positions in the array.
    The sample array is optionally rotated identically to preserve alignment.

    Parameters
    ----------
    trig_arr : ndarray of int (0 or 1), shape (N,)
        Circular array marking trigger positions.  Must contain one or more
        contiguous 1s; all other entries must be 0.
    samp_arr : ndarray, shape (N,)
        Sample values corresponding to each position in `trig_arr`.
    reorder_samples : bool
        Whether to reorder the sample array or keep it as originally presented

    Returns
    -------
    trig_reordered : ndarray of int, shape (N,)
        The trigger array rotated so that its contiguous 1s occupy the final
        positions of the array.
    samp_reordered : ndarray, shape (N,)
        The sample array, optionally rotated identically to `trig_arr` to maintain
        value-trigger correspondence.

    Raises
    ------
    ValueError
        If `trig_arr` and `samp_arr` have different lengths.
        If `trig_arr` does not contain any 1s.
        If the 1s in `trig_arr` are not contiguous.

    Examples
    --------
    >>> trig = np.array([0, 0, 1, 1, 0, 0])
    >>> samp = np.arange(6)
    >>> t_new, s_new = reorder_circular_samples_with_trigger(trig, samp)
    >>> t_new
    array([0, 0, 0, 0, 1, 1])
    >>> s_new
    array([4, 5, 0, 1, 2, 3])
    """
    # Basic validation
    if trig_arr.ndim != 1 or samp_arr.ndim != 1:
        raise ValueError("Both arrays must be 1D")
    if trig_arr.shape != samp_arr.shape:
        raise ValueError("trig_arr and samp_arr must have the same shape")

    n = trig_arr.size

    # Find indices of ones
    ones_idx = np.flatnonzero(trig_arr == 1)
    if ones_idx.size == 0:
        raise ValueError("trig_arr must contain at least one '1'")

    # Check contiguity (in circular sense)
    # Identify breaks in the sorted ones_idx sequence
    diffs = np.diff(ones_idx)
    breaks = np.nonzero(diffs != 1)[0]  # indices where run breaks (non-contiguous)

    # Check cases:
    #   1) no breaks  → single linear run
    #   2) one break AND wrap condition → circular (wrap) run
    if breaks.size == 0:
        # linear run
        start = ones_idx[0]
    elif breaks.size == 1 and ones_idx[0] == 0 and ones_idx[-1] == n - 1:
        # wrap-around run: pick the second run’s start
        brk = breaks[0]
        start = ones_idx[brk + 1]
    else:
        raise ValueError("1s must form exactly one contiguous block (possibly wrapping) in trig_arr")

    m = ones_idx.size
    # We want the block of length m to occupy indices [n-m ... n-1].
    # The element at 'start' should move to index n-m, so shift = (n-m) - start.
    shift = (n - m) - int(start)

    # Positive shift in np.roll moves elements right
    trig_reordered = np.roll(trig_arr, shift)
    if reorder_samples:
        samp_reordered = np.roll(samp_arr, shift)
        return trig_reordered, samp_reordered
    else:
        return trig_reordered, samp_arr
