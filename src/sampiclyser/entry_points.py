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

This module defines top-level functions that can be used as console_scripts
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
def print_channel_hits(
    decoded_file: Path,
    root_tree: str,
):
    """
    Print channel hit counts from a decoded SAMPIC run file.
    """

    hit_summary = sampiclyser.get_channel_hits(file_path=decoded_file, root_tree=root_tree)

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
    '--cms-label',
    'cms_label',
    type=str,
    default="Preliminary",
    help='The plot label to put near the CMS text. Default: Preliminary',
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
    cms_label: str,
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

    hit_summary = sampiclyser.get_channel_hits(file_path=decoded_file, root_tree=root_tree)

    fig = sampiclyser.plot_channel_hits(
        df=hit_summary,
        first_channel=first,
        last_channel=last,
        cms_label=cms_label,
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
    '--scale-factor',
    'scale_factor',
    type=float,
    default=1.0,
    help='The scale factor to apply to the hit count, used for adjusting central trigger. Default: 1.0',
)
@click.option(
    '--cms-label',
    'cms_label',
    type=str,
    default="Preliminary",
    help='The plot label to put near the CMS text. Default: Preliminary',
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
    scale_factor: float,
    cms_label: str,
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
        plot_hits=plot_hits,
        start_time=start_time,
        end_time=end_time,
        root_tree=root_tree,
        scale_factor=scale_factor,
        cms_label=cms_label,
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
    '--scale-factor',
    'scale_factor',
    type=float,
    default=1.0,
    help='The scale factor to apply to the hit count, used for adjusting central trigger. Default: 1.0',
)
@click.option(
    '--cms-label',
    'cms_label',
    type=str,
    default="Preliminary",
    help='The plot label to put near the CMS text. Default: Preliminary',
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
    scale_factor: float,
    cms_label: str,
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
        plot_hits=plot_hits,
        start_time=start_time,
        end_time=end_time,
        root_tree=root_tree,
        scale_factor=scale_factor,
        cms_label=cms_label,
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
