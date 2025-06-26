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

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

# import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import uproot
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
