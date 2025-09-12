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

from dataclasses import dataclass
from math import floor
from typing import Dict
from typing import Sequence
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from sampiclyser.sampic_tools import sampiclyser_style


@dataclass
class SensorSpec:
    """
    Specification for plotting a single sensor's hitmap, including local-to-global coordinate mapping.

    Supports three geometry types (grid, grouped, scatter) and optional global
    transformations (multiples of 90° rotations and an optional mirror flip).

    Attributes
    ----------
    name : str
        Human-readable title for the sensor (used as the subplot title).
    sampic_map : dict of int → int
        Mapping from SAMPIC channel indices to sensor channel identifiers.
    geometry : tuple
        Defines the sensor's layout and drawing method. The first element is
        the geometry type, one of:

        - `"grid"`
          A regular n_rows × n_cols grid with 1:1 channel→pixel mapping:
          (`"grid"`, n_rows, n_cols, chan2coord) where `chan2coord` maps
          sensor channel → (row, col) coordinates.
        - `"grouped"`
          Multiple pixels per channel on a grid:
          (`"grouped"`, chan2pixels, n_rows, n_cols) where `chan2pixels` maps
          sensor channel → list of (row, col) pixel coordinates.
        - `"scatter"`
          Arbitrary pixel centers and sizes:
          (`"scatter"`, chan2coords, pixel_width, pixel_height) where
          `chan2coords` maps sensor channel → (x, y) center coordinates.

    cmap : str
        Name of the Matplotlib colormap to use for this sensor
        (default: `"viridis"`).
    global_rotation_units : int
        Number of 90° clockwise rotations to apply to the local coordinate
        system to obtain the global orientation (0-3).
    global_flip : bool
        If True, mirror the coordinates horizontally *before* rotation to align
        the local layout with the global coordinate frame.
    """

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


# Possible sensors
spec1 = SensorSpec(
    name="Sensor A",
    sampic_map={0: 0, 1: 1, 2: 24},
    geometry=("grid", 5, 5, {i: (floor(i / 5), i % 5) for i in range(25)}),
    global_rotation_units=0,
    # global_flip=True,
)

spec2 = SensorSpec(
    name="Sensor B",
    sampic_map={0: 0, 4: 1, 5: 2},
    geometry=("grouped", {0: [(0, 0), (0, 1)], 1: [(1, 0), (1, 1)], 2: [(4, 4), (3, 3)]}, 5, 5),
    global_rotation_units=0,
    # global_flip=True,
)

spec3 = SensorSpec(
    name="Sensor C", sampic_map={0: "A", 1: "B", 2: "C"}, geometry=("scatter", {"A": (0.5, 0.5), "B": (1.5, 0.5), "C": (4.5, 4.5)}, 1, 1)
)


def convert_nrows_ncols_to_global(nrows: int, ncols: int, rotations: int) -> tuple[int, int]:
    """
    Compute the global grid dimensions after applying quarter-turn rotations.

    Parameters
    ----------
    nrows : int
        Number of rows in the local (unrotated) grid.
    ncols : int
        Number of columns in the local (unrotated) grid.
    rotations : int
        Number of 90° clockwise rotations to apply. Only the parity of
        `rotations` modulo 2 affects the shape: even → no swap,
        odd → swap rows and columns.

    Returns
    -------
    global_nrows : int
        Number of rows in the grid after rotation.
    global_ncols : int
        Number of columns in the grid after rotation.

    Examples
    --------
    >>> convert_nrows_ncols_to_global(10, 5, 0)
    (10, 5)
    >>> convert_nrows_ncols_to_global(10, 5, 1)
    (5, 10)
    >>> convert_nrows_ncols_to_global(10, 5, 2)
    (10, 5)
    >>> convert_nrows_ncols_to_global(10, 5, 3)
    (5, 10)
    """
    if rotations % 2 == 0:
        return nrows, ncols
    else:
        return ncols, nrows


def convert_r_c_to_global(r_local: int, c_local: int, rotations: int, do_mirror: bool, nrows: int, ncols: int) -> tuple[int, int]:
    """
    Map local (row, column) indices to global coordinates with optional mirroring and rotation.

    Parameters
    ----------
    r_local : int
        Row index in the sensor's local coordinate system (0-based).
    c_local : int
        Column index in the sensor's local coordinate system (0-based).
    rotations : int
        Number of 90° clockwise rotations to apply. Effective rotations are
        taken modulo 4 (i.e. 0, 1, 2, or 3).
    do_mirror : bool
        If True, reflect the column coordinate horizontally *before* rotation.
    nrows : int
        Total number of rows in the local grid.
    ncols : int
        Total number of columns in the local grid.

    Returns
    -------
    r_global, c_global : tuple of int
        The transformed (row, column) in the global coordinate system after
        applying mirroring and rotation.

    Notes
    -----
    - Mirroring (if enabled) flips `c_local` to `ncols - 1 - c_local`.
    - Rotation is performed clockwise in 90° increments:
      - 0 → (r, c)
      - 1 → (c, nrows - 1 - r)
      - 2 → (nrows - 1 - r, ncols - 1 - c)
      - 3 → (ncols - 1 - c, r)
    - Inputs outside expected ranges (e.g. negative indices) are not checked,
      so passing invalid `r_local`/`c_local` may lead to unexpected results.
    """
    # Apply mirror if requested
    r = r_local
    c = (ncols - 1 - c_local) if do_mirror else c_local

    # Normalize rotations into [0,3]
    rot = rotations % 4

    if rot == 0:
        return (r, c)
    if rot == 1:
        return (c, nrows - 1 - r)
    if rot == 2:
        return (nrows - 1 - r, ncols - 1 - c)
    if rot == 3:
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
    """
    Render a grid-based sensor hitmap on the given axes.

    Pixels with zero hits are left transparent; nonzero pixels are drawn
    as colored squares with a black border. Optionally, each cell can be
    annotated with its SAMPIC or board channel index in either the local
    or global coordinate system.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes into which to draw the grid hitmap.
    spec : SensorSpec
        Specification for this sensor, including:
        - `geometry = ("grid", nrows, ncols, chan2coord)`
          where `chan2coord` maps board channel → (row, col).
    hits_by_chan : dict of int → int
        Mapping from SAMPIC channel index to its total hit count.
    norm : matplotlib.colors.Normalize
        Normalization instance for mapping hit counts to the colormap.
    do_sampic_ch : bool, optional
        If True, annotate each cell with the SAMPIC channel index.
        Default is False.
    do_board_ch : bool, optional
        If True, annotate each cell with the board channel index.
        Default is False.
    center_fontsize : int, optional
        Font size (in points) for channel-number annotations at each cell center.
        Default is 14.
    coordinates : {'local', 'global'}, optional
        Coordinate system for plotting:
        - 'local': use the raw (row, col) as given in `chan2coord`.
        - 'global': apply `spec.global_rotation_units` and `spec.global_flip`
          to convert to the global frame.

    Returns
    -------
    im : matplotlib.image.AxesImage
        The AxesImage object returned by `ax.imshow`, for use with colorbars.

    Notes
    -----
    - Cells with zero hits are masked (transparent) to highlight active pixels.
    - The black border is drawn around every nonmasked cell for consistency
      with other sensor geometry plots.
    """
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
) -> None:
    """
    Render a grouped-pixel sensor hitmap with dual-level outlining and annotations.

    Draws each pixel in a group with a light-grey interior border, a single
    bold black outline around the group's bounding box, and optional
    channel-index labels at the group center.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to draw the hitmap.
    spec : SensorSpec
        Sensor specification with `geometry = ("grouped", chan2pixels, nrows, ncols)`,
        where `chan2pixels` maps board channel → list of (row, col) tuples.
    hits_by_chan : dict of int → int
        Mapping from SAMPIC channel index to its hit count.
    norm : matplotlib.colors.Normalize
        Normalization for mapping hit counts to colormap values.
    do_sampic_ch : bool, optional
        If True, annotate each group with the SAMPIC channel index.
    do_board_ch : bool, optional
        If True, annotate each group with the board channel index.
    center_fontsize : int, optional
        Font size for the annotation placed at the group's center
        (default: 14).
    coordinates : {'local', 'global'}, optional
        Coordinate system for pixel placement and annotation:
        - 'local': use raw (row, col) as given.
        - 'global': apply `spec.global_rotation_units` and `spec.global_flip`
          to convert to global coordinates.

    Returns
    -------
    None
        Draws directly onto `ax`; no return value.

    Notes
    -----
    - Colors are assigned per group via `ax.add_patch(Rectangle(..., facecolor=...) )`
      using `norm` and `spec.cmap`.
    - Internal pixel borders use a light-grey edge; the outer group border
      is drawn once around the minimal rectangle enclosing all member pixels.
    - Channel-index annotations are centered within the group's bounding box.
    """
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
) -> None:
    """
    Render an arbitrary-layout sensor hitmap using rectangular patches.

    Each sensor channel is represented by a rectangle of the specified
    width and height, colored according to its hit count and outlined
    with a black border. Optionally, channel indices are annotated at
    each rectangle's center.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to draw the hitmap.
    spec : SensorSpec
        Sensor specification with
        `geometry = ("scatter", chan2coords, pixel_width, pixel_height)`, where
        `chan2coords` maps board channel → (x, y) center coordinates, and
        `pixel_width`/`pixel_height` give the rectangle dimensions.
    hits_by_chan : dict of int → int
        Mapping from SAMPIC channel index to its hit count.
    norm : matplotlib.colors.Normalize
        Normalization instance for mapping hit counts to the colormap.
    do_sampic_ch : bool, optional
        If True, annotate each rectangle with the SAMPIC channel index.
    do_board_ch : bool, optional
        If True, annotate each rectangle with the board channel index.
    center_fontsize : int, optional
        Font size for annotations placed at each rectangle's center
        (default: 14).
    coordinates : {'local', 'global'}, optional, not implemented
        Coordinate system for rectangle placement and annotation:
        - 'local': use the raw (x, y) from `chan2coords`.
        - 'global': apply `spec.global_rotation_units` and `spec.global_flip`
          to convert to the global frame.

    Returns
    -------
    None
        Draws directly onto `ax`; no return value.

    Notes
    -----
    - Rectangles are created via `matplotlib.patches.Rectangle`, centered
      at (x, y) with width `pixel_width` and height `pixel_height`.
    - Colors are set by `cmap(norm(hit_count))`.
    - Black borders outline each rectangle uniformly.
    - Annotations (if any) are centered within each rectangle.
    """
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
    Draw a grid of sensor hitmaps with a shared color scale.

    Each sensor's hit counts are rendered according to its geometry
    (grid, grouped, or scatter) in a subplot arranged by `layout`.  All
    subplots share the same color normalization (linear or logarithmic),
    and have equal aspect ratio to preserve pixel shapes.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        DataFrame with columns `"Channel"` (int) and `"Hits"` (int).
    specs : sequence of SensorSpec
        Specifications for each sensor to plot (one subplot per spec).
    layout : tuple of int
        Subplot grid dimensions as (nrows, ncols).
    figsize : tuple of float, optional
        Figure size in inches as (width, height) (default: (8, 6)).
    cmap : str, optional
        Matplotlib colormap name to use for all sensors (default: "viridis").
    log_z : bool, optional
        If True, apply logarithmic normalization on the color (z) axis
        (default: False for linear scale).
    title : str or None, optional
        Overall figure title; if None, no supertitle is drawn (default: None).
    do_sampic_ch : bool, optional
        If True, annotate each pixel/group with its SAMPIC channel index
        (default: False).
    do_board_ch : bool, optional
        If True, annotate each pixel/group with its board channel index
        (default: False).
    center_fontsize : int, optional
        Font size for center annotations (default: 14).
    coordinates : {'local', 'global'}, optional
        Coordinate system for rendering:
        - 'local': use sensor's native coordinates.
        - 'global': apply `global_rotation_units` and `global_flip` from each spec
          to map to a common global frame (default: 'local').

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Figure object containing the arranged hitmap subplots.

    Notes
    -----
    - Uses sampiclyser_style for style formatting.
    - A single colorbar is added to the first subplot, reflecting all panels.
    - Subplot aspect is set to 'equal' so that pixels are not distorted.
    """
    hits_by_chan = dict(zip(summary_df["Channel"], summary_df["Hits"]))
    all_vals = np.array(list(hits_by_chan.values()), dtype=float)
    pos = all_vals[all_vals > 0]
    vmin = pos.min() if pos.size > 0 else 0
    vmax = all_vals.max() if all_vals.size > 0 else 1
    norm = LogNorm(vmin=vmin, vmax=vmax) if log_z else Normalize(vmin=0, vmax=vmax)

    with plt.style.use(sampiclyser_style):
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
