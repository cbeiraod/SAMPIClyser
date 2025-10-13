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
"""
Library-level entry points for command-line and programmatic access.

This module defines top-level functions that can be used as console scripts
entry points or imported directly into other Python code.
"""

import datetime
from pathlib import Path

import click
import matplotlib.pyplot as plt

import sampiclyser
import sampiclyser.sampic_convert_script


@click.group()
def cli() -> None:
    """SAMPIClyser command-line interface"""
    pass


cli.add_command(sampiclyser.sampic_convert_script.decode)


@cli.command()
def version():
    """Print the SAMPIClyser version"""
    print(f"The SAMPIClyser version is: {sampiclyser.__version__}")


@cli.command()
@click.argument('decoded_file', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--root-tree',
    'root_tree',
    type=str,
    default="sampic_hits",
    help='The name of the root ttree under which to save the hit data. Default: sampic_hits',
)
@click.option(
    '--batch-size',
    'batch_size',
    type=int,
    default=100000,
    help='Number of hits to read at once, as a batch, from disk. Default: 100 000. You should not need to tune this parameter unless in a memory constrained system or searching for ultimate performance.',
)
def print_channel_hits(
    decoded_file: Path,
    root_tree: str,
    batch_size: int,
):
    """
    Print channel hit counts from a decoded SAMPIC run file.
    """

    hit_summary = sampiclyser.get_channel_hits(file_path=decoded_file, batch_size=batch_size, root_tree=root_tree)

    click.echo(hit_summary)


@cli.command()
@click.argument('decoded_file', type=click.Path(exists=True, path_type=Path))
@click.option('--first', '-f', type=int, required=True, help='First channel to consider')
@click.option('--last', '-l', type=int, required=True, help='Last channel to consider')
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Path to save the plot')
@click.option(
    '--root-tree',
    'root_tree',
    type=str,
    default="sampic_hits",
    help='The name of the root ttree under which to save the hit data. Default: sampic_hits',
)
@click.option(
    '--batch-size',
    'batch_size',
    type=int,
    default=100000,
    help='Number of hits to read at once, as a batch, from disk. Default: 100 000. You should not need to tune this parameter unless in a memory constrained system or searching for ultimate performance.',
)
@click.option(
    '--label',
    'label',
    type=str,
    default="Preliminary",
    help='The plot label to put near the top left text. Default: Preliminary',
)
@click.option('--logy', 'log_y', is_flag=True, help='Enable logarithmic y axis')
@click.option(
    '--fig-width',
    'fig_width',
    type=float,
    default=15,
    help='The width of the plot. Default: 15',
)
@click.option(
    '--fig-height',
    'fig_height',
    type=float,
    default=9,
    help='The height of the plot. Default: 9',
)
@click.option(
    '--right-label',
    'rlabel',
    type=str,
    default="Test",
    help='The plot label to put on the right side of the figure, typically the beam details. Default: Test',
)
@click.option(
    '--is-data',
    '-d',
    'is_data',
    is_flag=True,
    help='Whether the processed data corresponds to real data (as an alternative to simulation data)',
)
@click.option(
    '--title',
    '-t',
    'title',
    type=str,
    default=None,
    help='The plot title to put at the top of the figure. Default: None',
)
def plot_hits(
    decoded_file: Path,
    first: int,
    last: int,
    output: Path,
    root_tree: str,
    batch_size: int,
    label: str,
    log_y: bool,
    fig_width: float,
    fig_height: float,
    rlabel: str,
    is_data: bool,
    title: str,
):
    """
    Plot channel hit counts from a decoded SAMPIC run file.
    """

    hit_summary = sampiclyser.get_channel_hits(file_path=decoded_file, batch_size=batch_size, root_tree=root_tree)

    fig = sampiclyser.plot_channel_hits(
        df=hit_summary,
        first_channel=first,
        last_channel=last,
        label=label,
        log_y=log_y,
        figsize=(fig_width, fig_height),
        rlabel=rlabel,
        is_data=is_data,
        title=title,
    )

    if output:
        fig.savefig(output)
    else:
        plt.show()


@cli.command()
@click.argument('decoded_file', type=click.Path(exists=True, path_type=Path))
@click.option('--bin-size', '-b', type=float, default=0.5, help='Bin size in seconds for hit-rate histogram. Default: 0.5 s')
@click.option('--output', '-o', type=click.Path(), help='Path to save the plot image')
@click.option('--plot-hits', 'plot_hits', is_flag=True, help='Enable hit per bin plotting instead of rate per bin')
@click.option(
    "--start-time",
    "start_time",
    type=click.DateTime(formats=["%Y-%m-%dT%H:%M:%S"]),
    help="Overrirde run start time in ISO format, e.g. '2025-06-20T17:45:32'.",
)
@click.option(
    "--end-time",
    "end_time",
    type=click.DateTime(formats=["%Y-%m-%dT%H:%M:%S"]),
    help="Overrirde run end time in ISO format, e.g. '2025-06-20T18:00:00'.",
)
@click.option(
    '--root-tree',
    'root_tree',
    type=str,
    default="sampic_hits",
    help='The name of the root ttree under which to save the hit data. Default: sampic_hits',
)
@click.option(
    '--batch-size',
    'batch_size',
    type=int,
    default=100000,
    help='Number of hits to read at once, as a batch, from disk. Default: 100 000. You should not need to tune this parameter unless in a memory constrained system or searching for ultimate performance.',
)
@click.option(
    '--scale-factor',
    'scale_factor',
    type=float,
    default=1.0,
    help='The scale factor to apply to the hit count, used for adjusting central trigger. Default: 1.0',
)
@click.option(
    '--label',
    'label',
    type=str,
    default="Preliminary",
    help='The plot label to put near the top left text. Default: Preliminary',
)
@click.option('--logy', 'log_y', is_flag=True, help='Enable logarithmic y axis')
@click.option(
    '--fig-width',
    'fig_width',
    type=float,
    default=15,
    help='The width of the plot. Default: 15',
)
@click.option(
    '--fig-height',
    'fig_height',
    type=float,
    default=9,
    help='The height of the plot. Default: 9',
)
@click.option(
    '--right-label',
    'rlabel',
    type=str,
    default="Test",
    help='The plot label to put on the right side of the figure, typically the beam details. Default: Test',
)
@click.option(
    '--is-data',
    '-d',
    'is_data',
    is_flag=True,
    help='Whether the processed data corresponds to real data (as an alternative to simulation data)',
)
@click.option(
    '--title',
    '-t',
    'title',
    type=str,
    default=None,
    help='The plot title to put at the top of the figure. Default: None',
)
def plot_hit_rate(
    decoded_file: Path,
    bin_size: float,
    output: Path,
    plot_hits: bool,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    root_tree: str,
    batch_size: int,
    scale_factor: float,
    label: str,
    log_y: bool,
    fig_width: float,
    fig_height: float,
    rlabel: str,
    is_data: bool,
    title: str,
):
    """
    Plot hit rate vs time from a decoded SAMPIC run file.
    """

    fig = sampiclyser.plot_hit_rate(
        file_path=decoded_file,
        bin_size=bin_size,
        batch_size=batch_size,
        plot_hits=plot_hits,
        start_time=start_time,
        end_time=end_time,
        root_tree=root_tree,
        scale_factor=scale_factor,
        label=label,
        log_y=log_y,
        figsize=(fig_width, fig_height),
        rlabel=rlabel,
        is_data=is_data,
        title=title,
    )

    if output:
        fig.savefig(output)
    else:
        plt.show()


@cli.command()
@click.argument('decoded_file', type=click.Path(exists=True, path_type=Path))
@click.option('--channel', '-c', type=int, default=0, help='The SAMPIC channel to draw. Default: 0')
@click.option('--bin-size', '-b', type=float, default=0.5, help='Bin size in seconds for hit-rate histogram. Default: 0.5 s')
@click.option('--output', '-o', type=click.Path(), help='Path to save the plot image')
@click.option('--plot-hits', 'plot_hits', is_flag=True, help='Enable hit per bin plotting instead of rate per bin')
@click.option(
    "--start-time",
    "start_time",
    type=click.DateTime(formats=["%Y-%m-%dT%H:%M:%S"]),
    help="Overrirde run start time in ISO format, e.g. '2025-06-20T17:45:32'.",
)
@click.option(
    "--end-time",
    "end_time",
    type=click.DateTime(formats=["%Y-%m-%dT%H:%M:%S"]),
    help="Overrirde run end time in ISO format, e.g. '2025-06-20T18:00:00'.",
)
@click.option(
    '--root-tree',
    'root_tree',
    type=str,
    default="sampic_hits",
    help='The name of the root ttree under which to save the hit data. Default: sampic_hits',
)
@click.option(
    '--batch-size',
    'batch_size',
    type=int,
    default=100000,
    help='Number of hits to read at once, as a batch, from disk. Default: 100 000. You should not need to tune this parameter unless in a memory constrained system or searching for ultimate performance.',
)
@click.option(
    '--scale-factor',
    'scale_factor',
    type=float,
    default=1.0,
    help='The scale factor to apply to the hit count, used for adjusting central trigger. Default: 1.0',
)
@click.option(
    '--label',
    'label',
    type=str,
    default="Preliminary",
    help='The plot label to put near the top left text. Default: Preliminary',
)
@click.option('--logy', 'log_y', is_flag=True, help='Enable logarithmic y axis')
@click.option(
    '--fig-width',
    'fig_width',
    type=float,
    default=15,
    help='The width of the plot. Default: 15',
)
@click.option(
    '--fig-height',
    'fig_height',
    type=float,
    default=9,
    help='The height of the plot. Default: 9',
)
@click.option(
    '--right-label',
    'rlabel',
    type=str,
    default="Test",
    help='The plot label to put on the right side of the figure, typically the beam details. Default: Test',
)
@click.option(
    '--is-data',
    '-d',
    'is_data',
    is_flag=True,
    help='Whether the processed data corresponds to real data (as an alternative to simulation data)',
)
@click.option(
    '--title',
    '-t',
    'title',
    type=str,
    default=None,
    help='The plot title to put at the top of the figure. Default: None',
)
def plot_channel_hit_rate(
    decoded_file: Path,
    channel: int,
    bin_size: float,
    output: Path,
    plot_hits: bool,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    root_tree: str,
    batch_size: int,
    scale_factor: float,
    label: str,
    log_y: bool,
    fig_width: float,
    fig_height: float,
    rlabel: str,
    is_data: bool,
    title: str,
):
    """
    Plot hit rate of a specific SAMPIC channel vs time from a decoded SAMPIC run file.
    """

    fig = sampiclyser.plot_channel_hit_rate(
        file_path=decoded_file,
        channel=channel,
        bin_size=bin_size,
        batch_size=batch_size,
        plot_hits=plot_hits,
        start_time=start_time,
        end_time=end_time,
        root_tree=root_tree,
        scale_factor=scale_factor,
        label=label,
        log_y=log_y,
        figsize=(fig_width, fig_height),
        rlabel=rlabel,
        is_data=is_data,
        title=title,
    )

    if output:
        fig.savefig(output)
    else:
        plt.show()


@cli.command()
@click.argument('decoded_file', type=click.Path(exists=True, path_type=Path))
@click.argument('config_file', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(), help='Path to save the plot image')
@click.option(
    '--root-tree',
    'root_tree',
    type=str,
    default="sampic_hits",
    help='The name of the root ttree under which to save the hit data. Default: sampic_hits',
)
@click.option(
    '--batch-size',
    'batch_size',
    type=int,
    default=100000,
    help='Number of hits to read at once, as a batch, from disk. Default: 100 000. You should not need to tune this parameter unless in a memory constrained system or searching for ultimate performance.',
)
@click.option(
    '--title',
    '-t',
    'title',
    type=str,
    default=None,
    help='The plot title to put at the top of the figure. Default: None',
)
@click.option(
    '--fig-width',
    'fig_width',
    type=float,
    default=15,
    help='The width of the plot. Default: 15',
)
@click.option(
    '--fig-height',
    'fig_height',
    type=float,
    default=9,
    help='The height of the plot. Default: 9',
)
@click.option(
    '--omit-sampic-ch',
    'omit_sampic_ch',
    is_flag=True,
    help='If set, the SAMPIC channel number will not be printed on top of the drawn channel in the hitmap.',
)
@click.option(
    '--omit-board-ch',
    'omit_board_ch',
    is_flag=True,
    help='If set, the SAMPIC channel number will not be printed on top of the drawn channel in the hitmap.',
)
@click.option(
    '--coordinates',
    'coordinates',
    type=str,
    default="local",
    help='Which coordinate system to use when drawing the hitmaps, valid options are local and global. Default: local',
)
@click.option('--logz', 'log_z', is_flag=True, help='Enable logarithmic z axis')
@click.option(
    '--color-map',
    'color_map',
    type=str,
    default="viridis",
    help='The color map to use for coloring the hit count, see matplotlib colormaps for valid options. Default: viridis',
)
@click.option(
    '--fontsize',
    'fontsize',
    type=float,
    default=14,
    help='The size of the font used to write channel numbers at the center of each pixel. Default: 14',
)
def plot_hitmap(
    decoded_file: Path,
    config_file: Path,
    output: Path,
    root_tree: str,
    batch_size: int,
    title: str,
    fig_width: float,
    fig_height: float,
    omit_sampic_ch: bool,
    omit_board_ch: bool,
    coordinates: str,
    log_z: bool,
    color_map: str,
    fontsize: float,
):
    """
    Plot a 2D hitmap of sensor hits from data.
    """
    from sampiclyser.sensor_hitmaps import SensorSpec

    extension = config_file.suffix.lower()
    if extension in [".json"]:
        import json

        with open(config_file, 'r') as infile:
            config_data = json.load(infile)
    elif extension in [".yml", ".yaml"]:
        import yaml

        with open(config_file, 'r') as infile:
            config_data = yaml.load(infile, yaml.Loader)
    else:
        print(f"Unknown config file extension: {extension}")

    layout = config_data["plot_layout"]
    sensor_types = config_data["sensor_types"]
    sensor_order = config_data["sensor_order"]
    sensors = config_data["sensor_specifications"]

    sensor_specs = []
    for sensor_name in sensor_order:
        if sensor_name not in sensors:
            raise RuntimeError(f"Sensor {sensor_name} defined in the order but not in the specifications")
        sensor_type = sensors[sensor_name]["sensor_type"]
        if sensor_type not in sensor_types:
            raise RuntimeError(f"Could not find the sensor type {sensor_type} in the list of defined sensor types")

        sensor_geometry = ()
        sensor_type = sensor_types[sensor_type]
        if sensor_type["geometry_type"] in ["grouped", "scatter"]:
            sensor_geometry = (sensor_type["geometry_type"], sensor_type['ch_to_coords'], sensor_type['rows'], sensor_type['cols'])
        elif sensor_type["geometry_type"] in ["grid"]:
            sensor_geometry = (sensor_type["geometry_type"], sensor_type['rows'], sensor_type['cols'], sensor_type['ch_to_coords'])
        else:
            raise RuntimeError(f"Unknown sensor spec type: {sensor_type['geometry_type']}")

        sensor_specs.append(
            SensorSpec(
                name=sensor_name,
                sampic_map=sensors[sensor_name]["datach_to_sensorch"],
                geometry=sensor_geometry,
                cmap=color_map,
                global_rotation_units=sensors[sensor_name]["global_90rotationUnits"],
                global_flip=sensors[sensor_name]["global_flip"],
            )
        )
        print(sensors[sensor_name]["global_90rotationUnits"])

    hit_summary = sampiclyser.get_channel_hits(file_path=decoded_file, batch_size=batch_size, root_tree=root_tree)

    fig = sampiclyser.plot_hitmap(
        summary_df=hit_summary,
        specs=sensor_specs,
        layout=layout,
        figsize=(fig_width, fig_height),
        cmap=color_map,
        log_z=log_z,
        title=title,
        do_sampic_ch=not omit_sampic_ch,
        do_board_ch=not omit_board_ch,
        center_fontsize=fontsize,
        coordinates=coordinates,
    )

    if output:
        fig.savefig(output)
    else:
        plt.show()


@cli.command()
@click.argument('decoded_file', type=click.Path(exists=True, path_type=Path))
@click.option('--channel', '-c', type=int, multiple=True, help='If set, only waveforms from the selected chanels will be plotted')
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Path to save the plot')
@click.option(
    '--root-tree',
    'root_tree',
    type=str,
    default="sampic_hits",
    help='The name of the root ttree under which to save the hit data. Default: sampic_hits',
)
@click.option(
    '--batch-size',
    'batch_size',
    type=int,
    default=100000,
    help='Number of hits to read at once, as a batch, from disk. Default: 100 000. You should not need to tune this parameter unless in a memory constrained system or searching for ultimate performance.',
)
@click.option(
    '--label',
    'label',
    type=str,
    default="Preliminary",
    help='The plot label to put near the top left text. Default: Preliminary',
)
@click.option('--logy', 'log_y', is_flag=True, help='Enable logarithmic y axis')
@click.option(
    '--fig-width',
    'fig_width',
    type=float,
    default=15,
    help='The width of the plot. Default: 15',
)
@click.option(
    '--fig-height',
    'fig_height',
    type=float,
    default=9,
    help='The height of the plot. Default: 9',
)
@click.option(
    '--right-label',
    'rlabel',
    type=str,
    default="Test",
    help='The plot label to put on the right side of the figure, typically the beam details. Default: Test',
)
@click.option(
    '--is-data',
    '-d',
    'is_data',
    is_flag=True,
    help='Whether the processed data corresponds to real data (as an alternative to simulation data)',
)
@click.option(
    '--title',
    '-t',
    'title',
    type=str,
    default=None,
    help='The plot title to put at the top of the figure. Default: None',
)
@click.option(
    '--skip-hits',
    'skip_hits',
    type=int,
    default=0,
    help='The number of hits to skip before starting to plot. Default: 0',
)
@click.option(
    '--num-hits',
    'num_hits',
    type=int,
    default=10,
    help='The number of hits to plot. Default: 10',
)
@click.option(
    '--name-override',
    'name_override',
    type=str,
    help='Name to use to override the file name',
)
@click.option(
    '--color-map',
    'color_map',
    type=str,
    default="tab10",
    help='The color map to use for coloring the hits, see matplotlib colormaps for valid options. Default: tab10',
)
@click.option(
    '--time-scale',
    'time_scale',
    type=click.Choice(['s', 'ms', 'us', 'ns', 'ps']),
    default="ns",
    help='The scale to use for the time axis. Default: ns',
)
def plot_waveforms(
    decoded_file: Path,
    channel,
    output: Path,
    root_tree: str,
    batch_size: int,
    label: str,
    log_y: bool,
    fig_width: float,
    fig_height: float,
    rlabel: str,
    is_data: bool,
    title: str,
    skip_hits: int,
    num_hits: int,
    name_override: str,
    color_map: str,
    time_scale: str,
):
    """
    Plot waveforms from a decoded SAMPIC run file, with some optional waveform selections.
    """
    time_scale_map = {
        "s": 1,
        "ms": 1e3,
        "us": 1e6,
        "ns": 1e9,
        "ps": 1e12,
    }

    fig = sampiclyser.plot_channel_waveforms(
        file_path=decoded_file,
        root_tree=root_tree,
        batch_size=batch_size,
        first_hit=skip_hits,
        num_hits=num_hits,
        channel_filter=[ch for ch in channel] if channel is not None and len(channel) > 0 else None,
        label=label,
        log_y=log_y,
        figsize=(fig_width, fig_height),
        rlabel=rlabel,
        is_data=is_data,
        title=title,
        file_name_id=name_override if name_override else None,
        cmap=color_map,
        time_scale=time_scale_map[time_scale],
    )

    if output:
        fig.savefig(output)
    else:
        plt.show()


@cli.command()
@click.argument('decoded_file', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--use-unix',
    'use_unix',
    is_flag=True,
    help='If set, use the unix timestamp for determining hit order instead of using SAMPIC reconstructed time',
)
@click.option('--find-all', 'find_all', is_flag=True, help='If set, will find all out of order events instead of stopping at the first one')
@click.option(
    '--root-tree',
    'root_tree',
    type=str,
    default="sampic_hits",
    help='The name of the root ttree under which to save the hit data. Default: sampic_hits',
)
@click.option(
    '--batch-size',
    'batch_size',
    type=int,
    default=100000,
    help='Number of hits to read at once, as a batch, from disk. Default: 100 000. You should not need to tune this parameter unless in a memory constrained system or searching for ultimate performance.',
)
def check_time_ordering(
    decoded_file: Path,
    use_unix: bool,
    find_all: bool,
    root_tree: str,
    batch_size: int,
):
    """
    Check time ordering of the hits from a decoded SAMPIC run file.
    """

    out_of_order = sampiclyser.check_time_ordering(
        decoded_file,
        use_unix_time=use_unix,
        find_all=find_all,
        batch_size=batch_size,
        root_tree=root_tree,
    )

    if len(out_of_order) > 0:
        if not find_all:
            raise RuntimeError("At least one hit is out of order")
        else:
            raise RuntimeError(f"Found {len(out_of_order)} hits out of order")
    else:
        print("All hits are ordered")
