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
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import uproot
from matplotlib.ticker import FormatStrFormatter


def get_channel_hits(file_path: Path, batch_size: int = 100_000, root_tree: str = "sampic_hits") -> pd.DataFrame:
    """
    Summarize hit counts per channel for a large Feather, Parquet, or ROOT file
    produced by the Sampic decoder.

    Reads only the 'Channel' column in fixed-size batches to limit memory use.

    Args:
        file_path: Path to the input .feather or .parquet file.
        batch_size: Number of rows to read per iteration.

    Returns:
        A DataFrame with columns:
          - “Channel”: the channel ID (int)
          - “Hits”:    total count of rows for that channel
        Sorted by increasing Channel.
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
        with open(file_path, "rb") as f:
            reader = ipc.open_file(f)
            for batch in reader.iter_batches(batch_size=batch_size):
                arr = batch.column("Channel").to_numpy()
                uniques, cnts = np.unique(arr, return_counts=True)
                for ch, cnt in zip(uniques, cnts):
                    counts[int(ch)] += int(cnt)

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
):
    """
    Plot a histogram of hit counts per channel in CMS style.

    Args:
        df (pd.DataFrame): Summary DataFrame with columns "Channel" and "Hits".
        first_channel (int): Lowest channel index to show.
        last_channel (int): Highest channel index to show.
        log_y (bool): If True, use a logarithmic y-axis.
        figsize (tuple): Figure size in inches as (width, height).
        rlabel (str): Text to display in the top-right corner (e.g., collision energy).
        is_data (bool): If True, the plot is labelled as data, if False the plot is labelled as simulation.
        color: Matplotlib color spec for the bars.
        title (str or None): Overall plot title, or None for no title.

    Channels outside df["Channel"] are shown with zero hits. Suppresses
    the scientific-offset power label on the y-axis for a cleaner look.
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
    plt.show()
