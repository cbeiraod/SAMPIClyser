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
SAMPIC conversion command-line interface.

This script processes raw SAMPIC binary data in a specified directory and
converts it into tabular formats: Parquet, Feather, and/or ROOT.  It streams
large files efficiently, supports hit-count limiting, and reports debugging
information when requested.

Usage example
-------------
python sampic_convert_script.py \
    --inputDir /path/to/binaries \
    --parquetFile out.parquet \
    --limitHits 100000
"""

from pathlib import Path

import click

import sampiclyser


@click.command()
@click.option(
    '--input-dir',
    '-i',
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help='Directory containing SAMPIC binary run files',
)
@click.option('--limit-hits', '-l', type=int, default=0, help='Maximum number of hits to process (0 = no limit)')
@click.option('--parquet', '-p', 'parquet_path', type=click.Path(path_type=Path), help='Output Parquet file')
@click.option('--feather', '-f', 'feather_path', type=click.Path(path_type=Path), help='Output Feather file')
@click.option('--root', '-r', 'root_path', type=click.Path(path_type=Path), help='Output ROOT file')
@click.option('--debug', '-d', is_flag=True, help='Enable debug logging')
@click.option(
    '--extra-header-bytes',
    'extra_header_bytes',
    type=int,
    default=1,
    help='Set the number of bytes to skip at the end of the header, by default only skip 1, to skip the newline character but in pathological cases it may need to be tuned',
)
@click.option(
    '--chunk-size',
    'chunk_size',
    type=int,
    default=64,
    help='Set how many bytes to load at a time when streaming data from the binary file, in units of 1kb. Default: 64. You should not need to tune this parameter unless in a memory constrained system or searching for ultimate performance.',
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
    help='Number of hits to collect before writing to disk. Default: 100 000. You should not need to tune this parameter unless in a memory constrained system or searching for ultimate performance.',
)
def decode(  # noqa: max-complexity=20
    input_dir: Path,
    limit_hits: int,
    parquet_path: Path | None,
    feather_path: Path | None,
    root_path: Path | None,
    debug: bool,
    extra_header_bytes: int,
    chunk_size: int,
    root_tree: str,
    batch_size: int,
):
    """
    Decode a raw SAMPIC run into Parquet, Feather, and/or ROOT formats.
    """

    if parquet_path is not None:
        if parquet_path.suffix not in ('.parquet', '.pq'):
            raise RuntimeError("The specified parquet file does not have the correct suffix")
        if parquet_path.exists():
            raise RuntimeError("The specified parquet file already exists")

    if feather_path is not None:
        if feather_path.suffix != ".feather":
            raise RuntimeError("The specified feather file does not have the correct suffix")
        if feather_path.exists():
            raise RuntimeError("The specified feather file already exists")

    if root_path is not None:
        if root_path.suffix != ".root":
            raise RuntimeError("The specified root file does not have the correct suffix")
        if root_path.exists():
            raise RuntimeError("The specified root file already exists")

    if debug:
        click.echo('Processing raw SAMPIC run directory: ', nl=False)
        click.secho(input_dir, fg='green', bold=True)

        if limit_hits > 0:
            click.echo('  - Will only process ', nl=False)
            click.secho(f'{limit_hits}', fg='red', bold=True, nl=False)
            click.echo(' hits')

        click.echo("  - Will add ", nl=False)
        click.secho(f'{extra_header_bytes}', fg='red', bold=True, nl=False)
        click.echo(f' byte{"s" if extra_header_bytes != 1 else ""} to the header')

        click.echo("  - Will process the data in  ", nl=False)
        click.secho(f'{chunk_size}', fg='red', bold=True, nl=False)
        click.echo('×1024 byte long chunks')

        if parquet_path is not None:
            click.echo('Saving parquet tabular data to ', nl=False)
            click.secho(parquet_path, fg='green', bold=True)
        if feather_path is not None:
            click.echo('Saving feather tabular data to ', nl=False)
            click.secho(feather_path, fg='green', bold=True)
        if root_path is not None:
            click.echo('Saving root tabular data to ', nl=False)
            click.secho(root_path, fg='green', bold=True)

    # Now that we finished loading the command line parameters, we can start processing the data
    decoder = sampiclyser.SAMPIC_Run_Decoder(input_dir)

    if debug:
        click.echo()
        click.echo(
            "The following binary files were found in the input and will be processed in this order. Also reporting the size of the headers for each:"
        )
        for file in decoder.run_files:
            with decoder.open_sampic_file_in_chunks_and_get_header(file, extra_header_bytes, chunk_size=chunk_size, debug=False) as (
                header,
                _,
            ):
                click.secho(file, fg='green')
                click.echo('\tHeader: ', nl=False)
                click.secho(len(header), nl=False, fg='blue')
                click.echo(' bytes')

    if debug:
        click.echo()
        click.echo("Processing and printing up to 4 hits to the terminal.")
        for raw_hit in decoder.parse_hit_records(limit_hits=4, extra_header_bytes=extra_header_bytes, chunk_size=chunk_size):
            click.echo(raw_hit)

    if feather_path is None and parquet_path is None and root_path is None:
        print("No output files were defined, not processing any data")
    else:
        decoder.decode_data(
            limit_hits=limit_hits,
            feather_path=feather_path,
            parquet_path=parquet_path,
            root_path=root_path,
            root_tree=root_tree,
            extra_header_bytes=extra_header_bytes,
            chunk_size=chunk_size,
            batch_size=batch_size,
        )


if __name__ == "__main__":
    decode()
