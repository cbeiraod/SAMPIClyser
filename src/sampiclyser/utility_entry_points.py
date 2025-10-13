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
Library-level entry points for command-line utilities.

This module defines top-level utility functions that can be used as console
scripts entry points or imported directly into other Python code.
"""

from pathlib import Path

import click

# import sampiclyser


@click.group()
def utilities() -> None:
    """SAMPIClyser command-line utilities interface"""
    pass


@utilities.command()
@click.argument('output_file', type=click.Path(exists=False, path_type=Path))
def generate_example_hitmap_config(output_file):
    """
    Generate an example configuration file describing the setup, necessary for hitmap plotting.
    Prefer using YAML files
    """
    example_file_data = dict(
        plot_layout=(1, 2),
        sensor_types=dict(
            type1_name=dict(
                geometry_type="grouped",
                rows=5,
                cols=5,
                ch_to_coords={
                    1: [(4, 0)],
                    2: [(4, 1), (4, 2), (3, 1), (3, 2)],
                    3: [(2, 0), (2, 1)],
                    4: [(1, 1)],
                    5: [(0, 0)],
                    6: [(0, 3), (1, 3), (2, 3)],
                    7: [(0, 4), (1, 4), (2, 4), (3, 4)],
                    8: [(2, 2)],
                    9: [(3, 3)],
                    10: [(4, 4)],
                },
            ),
        ),
        sensor_specifications=dict(
            sensor1_name=dict(
                datach_to_sensorch={0: 1, 1: 2, 2: 8, 6: 10, 7: 9},
                sensor_type="type1_name",
                global_90rotationUnits=2,
                global_flip=False,
            ),
            sensor2_name=dict(
                datach_to_sensorch={3: 8, 4: 2, 5: 1, 8: 9, 9: 10},
                sensor_type="type1_name",
                global_90rotationUnits=1,
                global_flip=True,
            ),
        ),
        sensor_order=["sensor1_name", "sensor2_name"],
    )

    extension = output_file.suffix.lower()
    if extension in [".json"]:
        import json

        with open(output_file, 'w') as outfile:
            json.dump(example_file_data, outfile, indent=2)
    elif extension in [".yml", ".yaml"]:
        import yaml

        with open(output_file, 'w') as outfile:
            yaml.dump(example_file_data, outfile, default_flow_style=False)
    else:
        print(f"Unknown config file extension: {extension}")


# TODO: Add an interactive config file generation script
