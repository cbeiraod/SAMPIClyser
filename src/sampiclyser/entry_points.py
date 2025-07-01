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
