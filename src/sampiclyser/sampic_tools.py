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
    suffix = file_path.suffix.lower()

    if suffix in (".parquet", ".pq"):
        # Parquet: iterate row‐group batches of just the Channel column
        pqf = pq.ParquetFile(str(file_path))
        for batch in pqf.iter_batches(batch_size=batch_size, columns=["Channel"]):
            arr = batch.column("Channel").to_numpy()
            uniques, cnts = np.unique(arr, return_counts=True)
            for ch, cnt in zip(uniques, cnts):
                counts[int(ch)] += int(cnt)

    elif suffix == ".feather":
        # Feather (Arrow IPC): open and iterate record batches
        dataset = ds.dataset(str(file_path), format="feather")
        scanner = dataset.scanner(batch_size=batch_size, columns=["Channel"])
        for batch in scanner.to_batches():
            arr = batch["Channel"].to_numpy()
            uniques, cnts = np.unique(arr, return_counts=True)
            for ch, cnt in zip(uniques, cnts):
                counts[int(ch)] += int(cnt)

        # with open(file_path, "rb") as f:
        #     reader = ipc.open_file(f)
        #     for i in range(reader.num_record_batches):
        #         batch = reader.get_batch(i)
        #         arr = batch.column("Channel").to_numpy()
        #         uniques, cnts = np.unique(arr, return_counts=True)
        #         for ch, cnt in zip(uniques, cnts):
        #             counts[int(ch)] += int(cnt)

    elif suffix == ".root":
        # ROOT: use uproot.iterate to stream the 'Channel' branch
        tree_path = f"{file_path}:{root_tree}"
        for batch in uproot.iterate(tree_path, ["Channel"], step_size=batch_size):
            arr = batch["Channel"]
            uniques, cnts = np.unique(arr, return_counts=True)
            for ch, cnt in zip(uniques, cnts):
                counts[int(ch)] += int(cnt)

    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

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
    suffix = file_path.suffix.lower()

    if suffix in (".parquet", ".pq"):
        pqf = pq.ParquetFile(str(file_path))
        for batch in pqf.iter_batches(batch_size=batch_size, columns=["UnixTime"]):
            arr = batch.column("UnixTime").to_numpy()
            for t in arr:
                idx = int((t - run_start_ts) // bin_size)
                if idx >= 0:
                    counts[idx] += 1

    elif suffix == ".feather":
        dataset = ds.dataset(str(file_path), format="feather")
        scanner = dataset.scanner(batch_size=batch_size, columns=["UnixTime"])
        for batch in scanner.to_batches():
            arr = batch["UnixTime"].to_numpy()
            for t in arr:
                idx = int((t - run_start_ts) // bin_size)
                if idx >= 0:
                    counts[idx] += 1

        # with open(file_path, "rb") as f:
        #     reader = ipc.open_file(f)
        #     for i in range(reader.num_record_batches):
        #         batch = reader.get_batch(i)
        #         arr = batch.column("UnixTime").to_numpy()
        #         for t in arr:
        #             idx = int((t - start_ts) // bin_size)
        #             if idx >= 0:
        #                 counts[idx] += 1

    elif suffix == ".root":
        tree_path = f"{file_path}:{root_tree}"
        for batch in uproot.iterate(tree_path, ["UnixTime"], step_size=batch_size):
            arr = batch["UnixTime"]
            for t in arr:
                idx = int((t - run_start_ts) // bin_size)
                if idx >= 0:
                    counts[idx] += 1

    else:
        raise ValueError(f"Unsupported format: {file_path.suffix}")

    if not counts:
        raise RuntimeError("No hits found in file.")

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
    ax.step(dtimes, rates, where="post", color=color)

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
