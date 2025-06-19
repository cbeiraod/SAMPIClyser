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
"""The command line script for converting binary data with SAMPIClyser
"""

import argparse
import pprint
from pathlib import Path

import sampiclyser


def main(args=None):
    """This is the entry function for the command line interface to print out the version"""
    parser = argparse.ArgumentParser(
        description='Process the SAMPIC binary data and store into ROOT/Feather/Parquet format. Parquet format is recommended.'
    )

    parser.add_argument(
        '-i',
        '--inputDir',
        metavar='DIR',
        help="The input directory with the binary data",
        required=True,
    )
    parser.add_argument(
        '--limitHits',
        metavar='INT',
        help="Limit data processing to this amount of hits. If not set, the default behaviour is to process the whole file",
    )
    parser.add_argument(
        '-p',
        '--parquetFile',
        metavar='File',
        help="The file to store the parquet output, if not set, there will be no parquet output",
    )
    parser.add_argument(
        '-f',
        '--featherFile',
        metavar='File',
        help="The file to store the feather output, if not set, there will be no feather output",
    )
    parser.add_argument(
        '-r',
        '--rootFile',
        metavar='File',
        help="The file to store the root output, if not set, there will be no ROOT output",
    )
    parser.add_argument(
        '-d',
        '--debug',
        help="If set, debug output will be enabled",
        action='store_true',
    )
    parser.add_argument(
        '--extraHeaderBytes',
        metavar='INT',
        help="Set the number of bytes to skip at the end of the header, by default only skip 1, to skip the newline character but in pathological cases it may need to be tuned",
        default='1',
    )
    parser.add_argument(
        '--chunkSize',
        metavar='INT',
        help="Set how many bytes to load at a time when streaming data from the binary file, in units of 1kb. Default: 64. You should not need to tune this parameter unless in a memory constrained system or searching for ultimate performance.",
        default='64',
    )

    args = parser.parse_args(args=args)

    inputDirectory = Path(args.inputDir)
    if not inputDirectory.exists() or not inputDirectory.is_dir():
        raise RuntimeError("The input directory does not exist or is not a directory, please fix")

    limitHits = 0
    if args.limitHits is not None:
        limitHits = int(args.limitHits, base=0)

    extraHeaderBytes = int(args.extraHeaderBytes, base=0)
    chunkSize = int(args.chunkSize, base=0) * 1024

    parquetFilePath = None
    if args.parquetFile is not None:
        parquetFilePath = Path(args.parquetFile)
        if parquetFilePath.suffix != ".parquet":
            raise RuntimeError("The specified parquet file does not have the correct suffix")
        if parquetFilePath.exists():
            raise RuntimeError("The specified parquet file already exists")

    featherFilePath = None
    if args.featherFile is not None:
        featherFilePath = Path(args.featherFile)
        if featherFilePath.suffix != ".feather":
            raise RuntimeError("The specified feather file does not have the correct suffix")
        if featherFilePath.exists():
            raise RuntimeError("The specified feather file already exists")

    rootFilePath = None
    if args.rootFile is not None:
        rootFilePath = Path(args.rootFile)
        if rootFilePath.suffix != ".root":
            raise RuntimeError("The specified root file does not have the correct suffix")
        if rootFilePath.exists():
            raise RuntimeError("The specified root file already exists")

    # Now that we finished loading the command line parameters, we can start processing the data
    decoder = sampiclyser.SAMPIC_Run_Decoder(inputDirectory)

    if args.debug:
        print()
        print(
            "The following binary files were found in the input and will be processed in this order. Also reporting the size of the headers for each:"
        )
        # print(decoder.run_files)
        for file in decoder.run_files:
            with decoder.open_sampic_file_in_chunks_and_get_header(file, extraHeaderBytes, chunk_size=chunkSize, debug=False) as (
                header,
                _,
            ):
                print(f"{file}:\n\tHeader: {len(header)} bytes")

    if args.debug:
        print()
        print("Processing and printing 4 hits to the terminal.")
        for raw_hit in decoder.parse_hit_records(limit_hits=4, extra_header_bytes=extraHeaderBytes, chunk_size=chunkSize):
            pprint.pp(raw_hit)

    if featherFilePath is None and parquetFilePath is None and rootFilePath is None:
        print("No output files were defined, not processing any data")
    else:
        decoder.decode_data(
            limit_hits=limitHits,
            feather_path=featherFilePath,
            parquet_path=parquetFilePath,
            root_path=rootFilePath,
            extra_header_bytes=extraHeaderBytes,
            chunk_size=chunkSize,
        )


if __name__ == "__main__":
    main()
