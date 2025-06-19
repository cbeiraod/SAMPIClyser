from dataclasses import dataclass
from typing import Dict
from typing import Sequence
from typing import Tuple

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


@dataclass
class SensorSpec:
    name: str
    sampic_map: Dict[int, int]
    geometry: Tuple  # e.g., ("grid", nrows, ncols, chan2coord),
    #       ("grouped", chan2pixels, nrows, ncols),
    #       ("scatter", chan2coords, pixel_width, pixel_height): TODO: make it respect the global transformation
    cmap: str = "viridis"
    # Information on how to relate local coordinates to global coordinates, a
    # simplified model only allowing multiples of 90 degree rotations and
    # a possible mirroring of coordinates is assumed
    global_rotation_units: int = 0  # how many 90 degree rotations
    global_flip: bool = False


def convert_nrows_ncols_to_global(nrows: int, ncols: int, rotations: int):
    if rotations % 2 == 0:
        return nrows, ncols
    else:
        return ncols, nrows


def convert_r_c_to_global(r_local: int, c_local: int, rotations: int, do_mirror: bool, nrows: int, ncols: int):
    r = r_local
    if do_mirror:
        c = ncols - 1 - c_local
    else:
        c = c_local

    if rotations == 0:
        return (r, c)
    if rotations == 1:
        return (c, nrows - 1 - r)
    if rotations == 2:
        return (nrows - 1 - r, ncols - 1 - c)
    if rotations == 3:
        return (ncols - 1 - c, r)


def _plot_grid_sensor(
    ax: plt.Axes,
    spec: SensorSpec,
    hits_by_chan: Dict[int, int],
    norm: Normalize,
    do_sampic_ch: bool = False,
    do_board_ch: bool = False,
    center_fontsize: int = 14,
    coordinates: str = "local",
):
    """Plot a 2D grid sensor, masking out zero-hit pixels, drawing borders,
    and annotating each pixel with its board channel."""
    _, nrows, ncols, chan2coord = spec.geometry
    rotations = spec.global_rotation_units % 4
    do_mirror = spec.global_flip

    if coordinates == "global":
        nrows_g, ncols_g = convert_nrows_ncols_to_global(nrows, ncols, rotations)
    else:
        nrows_g = nrows
        ncols_g = ncols

    arr = np.zeros((nrows_g, ncols_g), dtype=float)
    for samp_ch, sens_ch in spec.sampic_map.items():
        r_local, c_local = chan2coord[sens_ch]
        if coordinates == "global":
            r, c = convert_r_c_to_global(r_local, c_local, rotations, do_mirror, nrows, ncols)
        else:
            r = r_local
            c = c_local
        arr[r, c] = hits_by_chan.get(samp_ch, 0)

    masked = np.ma.masked_where(arr == 0, arr)
    im = ax.imshow(masked, interpolation="nearest", cmap=spec.cmap, origin="lower", norm=norm, extent=[0, ncols_g, 0, nrows_g])
    # draw borders and annotate
    for samp_ch, sens_ch in spec.sampic_map.items():
        r_local, c_local = chan2coord[sens_ch]
        if coordinates == "global":
            r, c = convert_r_c_to_global(r_local, c_local, rotations, do_mirror, nrows, ncols)
        else:
            r = r_local
            c = c_local
        if arr[r, c] > 0:
            ax.add_patch(Rectangle((c, r), 1, 1, fill=False, edgecolor="black", linewidth=0.5))
            if do_sampic_ch and do_board_ch:
                ax.text(c + 0.5, r + 0.5, f"{samp_ch}→{sens_ch}", ha='center', va='center', fontsize=center_fontsize, color='grey')
            elif do_sampic_ch:
                ax.text(c + 0.5, r + 0.5, str(samp_ch), ha='center', va='center', fontsize=center_fontsize, color='grey')
            elif do_board_ch:
                ax.text(c + 0.5, r + 0.5, str(sens_ch), ha='center', va='center', fontsize=center_fontsize, color='grey')
    ax.set_xlim(0, ncols_g)
    ax.set_ylim(0, nrows_g)
    ax.set_aspect('equal')
    return im


def _plot_grouped_sensor(
    ax: plt.Axes,
    spec: SensorSpec,
    hits_by_chan: Dict[int, int],
    norm: Normalize,
    do_sampic_ch: bool = False,
    do_board_ch: bool = False,
    center_fontsize: int = 14,
    coordinates: str = "local",
):
    """Plot a grouped-pixel sensor with light-grey internal borders,
    black outline around each group, and annotate group center with board channel."""
    _, chan2pixels, nrows, ncols = spec.geometry
    rotations = spec.global_rotation_units % 4
    do_mirror = spec.global_flip

    if coordinates == "global":
        nrows_g, ncols_g = convert_nrows_ncols_to_global(nrows, ncols, rotations)
    else:
        nrows_g = nrows
        ncols_g = ncols

    for samp_ch, sens_ch in spec.sampic_map.items():
        count = hits_by_chan.get(samp_ch, 0)
        pixels = chan2pixels[sens_ch]
        # draw internal pixels
        min_r = None
        max_r = None
        min_c = None
        max_c = None
        for r_local, c_local in pixels:
            if coordinates == "global":
                r, c = convert_r_c_to_global(r_local, c_local, rotations, do_mirror, nrows, ncols)
            else:
                r = r_local
                c = c_local
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor=plt.get_cmap(spec.cmap)(norm(count)), edgecolor="grey", linewidth=0.5))

            if min_r is None:
                min_r = r
                max_r = r
                min_c = c
                max_c = c
            else:
                if r < min_r:
                    min_r = r
                if r > max_r:
                    max_r = r
                if c < min_c:
                    min_c = c
                if c > max_c:
                    max_c = c
        # draw outer bounding box
        ax.add_patch(Rectangle((min_c, min_r), max_c - min_c + 1, max_r - min_r + 1, fill=False, edgecolor="black", linewidth=0.5))
        # annotate at center of group
        center_r = min_r + (max_r - min_r + 1) / 2
        center_c = min_c + (max_c - min_c + 1) / 2
        if do_sampic_ch and do_board_ch:
            ax.text(center_c, center_r, f"{samp_ch}→{sens_ch}", ha='center', va='center', fontsize=center_fontsize, color='grey')
        elif do_sampic_ch:
            ax.text(center_c, center_r, str(samp_ch), ha='center', va='center', fontsize=center_fontsize, color='grey')
        elif do_board_ch:
            ax.text(center_c, center_r, str(sens_ch), ha='center', va='center', fontsize=center_fontsize, color='grey')
    ax.set_xlim(0, ncols_g)
    ax.set_ylim(0, nrows_g)
    ax.set_aspect('equal')
    return None


def _plot_scatter_sensor(
    ax: plt.Axes,
    spec: SensorSpec,
    hits_by_chan: Dict[int, int],
    norm: Normalize,
    do_sampic_ch: bool = False,
    do_board_ch: bool = False,
    center_fontsize: int = 14,
    coordinates: str = "local",
):
    """Plot arbitrary sensor layout by drawing each pixel as rectangle with borders
    and annotating each with board channel."""
    _, chan2coords, pixel_width, pixel_height = spec.geometry
    xs, ys = [], []
    for samp_ch, sens_ch in spec.sampic_map.items():
        count = hits_by_chan.get(samp_ch, 0)
        x_center, y_center = chan2coords[sens_ch]
        ax.add_patch(
            Rectangle(
                (x_center - pixel_width / 2, y_center - pixel_height / 2),
                pixel_width,
                pixel_height,
                facecolor=plt.get_cmap(spec.cmap)(norm(count)),
                edgecolor="black",
                linewidth=0.5,
            )
        )
        # annotate center
        if do_sampic_ch and do_board_ch:
            ax.text(x_center, y_center, f"{samp_ch}→{sens_ch}", ha='center', va='center', fontsize=center_fontsize, color='grey')
        elif do_sampic_ch:
            ax.text(x_center, y_center, str(samp_ch), ha='center', va='center', fontsize=center_fontsize, color='grey')
        elif do_board_ch:
            ax.text(x_center, y_center, str(sens_ch), ha='center', va='center', fontsize=center_fontsize, color='grey')
        xs.append(x_center)
        ys.append(y_center)
    x_min, x_max = min(xs) - pixel_width / 2, max(xs) + pixel_width / 2
    y_min, y_max = min(ys) - pixel_height / 2, max(ys) + pixel_height / 2
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    return None


def plot_hitmap(
    summary_df: pd.DataFrame,
    specs: Sequence[SensorSpec],
    layout: Tuple[int, int],
    figsize: Tuple[int, float] = (8, 6),
    cmap: str = "viridis",
    log_z: bool = False,
    title: str | None = None,
    do_sampic_ch: bool = False,
    do_board_ch: bool = False,
    center_fontsize: int = 14,
    coordinates: str = "local",
) -> plt.Figure:
    """
    Draw a 2D hitmap for each sensor in `specs`, arranged in a grid,
    using a shared color scale (linear or log), and equal aspect.

    Args:
        summary_df: DataFrame with columns "Channel" and "Hits".
        specs:      List of SensorSpec, one per subplot.
        layout:     (nrows, ncols) grid layout.
        figsize:    Figure size in inches.
        cmap:       Default colormap name.
        log_z:      If True, apply logarithmic normalization on z-axis.
        title:      Optional overall figure title.
        do_sampic_ch: If True, draw the sampic channel at the center of the relevant pixels
        do_board_ch:  If True, draw the board channel at the center of the relevant pixels
        coordinates: The coordinate system to use for drawing the sensors, if local, then the local sensor coordinates are used, if global then a donwstream (looking down the beam) coordinate system is used.

    Returns:
        Matplotlib Figure with the hitmaps.
    """
    hits_by_chan = dict(zip(summary_df["Channel"], summary_df["Hits"]))
    all_vals = np.array(list(hits_by_chan.values()), dtype=float)
    pos = all_vals[all_vals > 0]
    vmin = pos.min() if pos.size > 0 else 0
    vmax = all_vals.max() if all_vals.size > 0 else 1
    norm = LogNorm(vmin=vmin, vmax=vmax) if log_z else Normalize(vmin=0, vmax=vmax)

    plt.style.use(hep.style.CMS)
    nrows, ncols = layout
    # Create figure and axes first, then apply title
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, constrained_layout=True)
    if title:
        fig.suptitle(title, weight='bold')

    if coordinates == "local":
        fig.text(
            1,
            1,  # (x, y) in figure coords
            "(Local sensor coordinates)",  # the string
            ha='right',  # align right edge
            va='top',  # align top edge
            transform=fig.transFigure,  # figure coordinate system
            fontsize=10,
        )
    elif coordinates == "global":
        fig.text(
            1,
            1,  # (x, y) in figure coords
            "(Downstream global coordinates)",  # the string
            ha='right',  # align right edge
            va='top',  # align top edge
            transform=fig.transFigure,  # figure coordinate system
            fontsize=10,
        )
    else:
        raise ValueError(f"Unknown coordinate system {coordinates}")

    first_img = None
    for ax, spec in zip(axes.flat, specs):
        geom = spec.geometry[0]
        if geom == "grid":
            im = _plot_grid_sensor(
                ax,
                spec,
                hits_by_chan,
                norm,
                do_sampic_ch=do_sampic_ch,
                do_board_ch=do_board_ch,
                center_fontsize=center_fontsize,
                coordinates=coordinates,
            )
        elif geom == "grouped":
            im = _plot_grouped_sensor(
                ax,
                spec,
                hits_by_chan,
                norm,
                do_sampic_ch=do_sampic_ch,
                do_board_ch=do_board_ch,
                center_fontsize=center_fontsize,
                coordinates=coordinates,
            )
        elif geom == "scatter":
            im = _plot_scatter_sensor(
                ax,
                spec,
                hits_by_chan,
                norm,
                do_sampic_ch=do_sampic_ch,
                do_board_ch=do_board_ch,
                center_fontsize=center_fontsize,
                coordinates=coordinates,
            )
        else:
            raise ValueError(f"Unknown geometry {geom}")
        ax.set_title(spec.name, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        if first_img is None and im is not None:
            first_img = im

    for ax in axes.flat[len(specs) :]:
        ax.axis("off")
    if first_img:
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        fig.colorbar(mappable, ax=axes, orientation='vertical', label='Hits')
        # fig.colorbar(first_img, ax=axes, orientation="vertical", label="Hits")
    else:
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        fig.colorbar(mappable, ax=axes, orientation='vertical', label='Hits')
    return fig
