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

import click

import sampiclyser.sampic_convert_script
from sampiclyser import __version__


@click.group()
def cli() -> None:
    """SAMPIClyser command-line interface"""
    pass


cli.add_command(sampiclyser.sampic_convert_script.decode)


@cli.command()
def version():
    """This is the entry function for the command line interface to print out the version"""

    print(f"The SAMPIClyser version is: {__version__}")
