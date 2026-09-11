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

import mmap
import re
import statistics
import struct
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from math import floor
from pathlib import Path
from struct import Struct
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import uproot
from natsort import natsorted
from pyarrow.ipc import new_file
from termcolor import colored

# SchemaInfo
SAMPIC_Schema_Info = {
    # Format:
    # Name: (pandas, pyarrow, numpy for root, optional array size)
    # Coordination
    "HitNumber": ("int32", pa.int32(), np.int32),
    # From SAMPIC
    "Channel": ("uint8", pa.uint8(), np.uint8),
    "FirstSampleTime_in_ps": ("int64", pa.int64(), np.int64),
    "FirstSampleTime_in_ps_fine": ("float64", pa.float64(), np.double),
    "RawTOTValue": ("int32", pa.int32(), np.int32),
    "TOTValue": ("int32", pa.int32(), np.int32),
    # From SAMPIC: Samples / Waveform
    "DataSize": ("int32", pa.int32(), np.int32),
    "DataSample": (None, pa.list_(pa.float32()), np.float32, 64),  # This has Time-INL applied to correct sampling
    # Measurements
    "Time": ("float64", pa.float64(), np.double),  # From Ordered+CFD (CFD extracted from the waveform)
    "Baseline": ("float32", pa.float32(), np.float32),
    "RawPeak": ("float32", pa.float32(), np.float32),
    "Amplitude": ("float32", pa.float32(), np.float32),
    # Other
    "FirstCellIndex": ("uint8", pa.uint8(), np.uint8),
    "NumTriggerSamples": ("uint8", pa.uint8(), np.uint8),
    "UnixTime": ("float64", pa.float64(), np.double),  # when received in computer
    # "Cell": ("int32", pa.int32(), np.int32), # Physical cell of first sample
    "TimeStampA": ("int32", pa.int32(), np.int32),
    "TimeStampB": ("int32", pa.int32(), np.int32),
    "FPGATimeStamp": ("uint64", pa.uint64(), np.uint64),
    "StartOfADCRamp": ("int32", pa.int32(), np.int32),
    "PhysicalCell0Time": ("float64", pa.float64(), np.double),
    "OrderedCell0Time": ("float64", pa.float64(), np.double),  # timestamp of first sample in ns from run start
    "Cell0Time": ("float64", pa.float64(), np.double),  # is this the same as above?
    "ADCCounterLatched": ("int32", pa.int32(), np.int32),
    "FirstSampleTimeStamp": ("float64", pa.float64(), np.double),
    # "TriggerPosition": (None, pa.list_(pa.int32()), np.int32, 64), # last samples may be affected by disconnecting, so need to be corrected
    # … etc …
}

# The maximum nanosecond timestamp before IEEE 754 64-bit floats
# lose 1 picosecond resolution (ULP jumps from ~0.97 ps to ~1.95 ps).
MAX_1PS_RESOLUTION_NS = 2.0**43


def calculate_wrap_offset_ps(bit_depth: int, freq_hz: int) -> int:
    """
    Calculates the exact time offset in picoseconds for a single hardware counter wrap.

    Args:
        bit_depth: The number of bits of the coarse counter (e.g., 40)
        freq_hz: The frequency of the coarse clock in Hz (e.g., 100_000_000)

    Returns:
        The exact offset in picoseconds as a Python integer.
    """
    # Total number of ticks before the counter rolls over to 0
    total_ticks = 1 << bit_depth

    # 1 second = 1,000,000,000,000 picoseconds
    PICOSECONDS_PER_SECOND = 1_000_000_000_000

    # Multiply first, then divide to prevent floating point conversion and loss of precision
    offset_ps = (total_ticks * PICOSECONDS_PER_SECOND) // freq_hz

    # Sanity check: Ensure the frequency divides cleanly into picoseconds
    if (total_ticks * PICOSECONDS_PER_SECOND) % freq_hz != 0:
        print(
            colored("Warning:", "yellow"),
            "Clock frequency does not map perfectly to integer picoseconds - A fractional picosecond of drift will accumulate on every wrap.",
        )

    return offset_ps


def convert_daq_ns_double_to_ps_int(daq_time_ns: float, wrap_count: int, max_counter: int):
    """
    Safely converts DAQ nanosecond floats to absolute int64 picoseconds.

    Args:
        daq_time_ns: float (from the DAQ, which is actually a double)
        wrap_count: int (how many times the counter rolled over)
        max_counter: int (the maximum number of values the counter counts before it wraps)
    """

    # The following sequence is important to be done in this way to avoid any rounding from floating point precision
    # 1 - Extract the whole nanoseconds (integer part)
    # Using floor ensures we safely grab the integer part without float rounding issues.
    hit_time_ns = floor(daq_time_ns)

    # 2 - Extract the exact fractional nanoseconds and convert to ps
    hit_time_remainder_ps = (daq_time_ns - hit_time_ns) * 1000

    # 3 - Extract integer picoseconds from the remainder
    hit_time_ps = floor(hit_time_remainder_ps)

    # 4 - Extract the sub-picosecond remainder
    hit_time_remainder_ps = hit_time_remainder_ps - hit_time_ps

    # 5 - Combine the integers into a base picosecond timestamp
    hit_time_ps += hit_time_ns * 1000

    # 6 - Adjust hit time for the number of wraps
    hit_time_ps = hit_time_ps + wrap_count * max_counter

    return hit_time_ps, hit_time_remainder_ps


def build_schema(
    columns: Union[List[str], pd.Index],
    metadata: Optional[Dict[bytes, bytes]] = None,
    schemaInfo: Dict[str, Tuple] = SAMPIC_Schema_Info,
) -> pa.Schema:
    """
    Construct a PyArrow Schema from predefined field types and optional metadata.

    This utility builds a fully-specified Arrow Schema by iterating over
    a `schemaInfo` mapping of field names to type definitions.  It
    includes only those fields whose PyArrow type is non-null, and
    attaches any provided metadata as key/value byte pairs.

    Parameters
    ----------
    metadata : dict of bytes->bytes, optional
        Key/value metadata to attach to the schema (e.g. from file header).
        If None, no metadata will be set on the schema.
    schemaInfo : dict of str->tuple
        Mapping from field name to a tuple defining:
          - pandas dtype string (unused here)
          - PyArrow DataType (used)
          - NumPy dtype for ROOT output
          - optional list/array size for array fields
        Only entries where the PyArrow DataType is not None are used.

    Returns
    -------
    pa.Schema
        A PyArrow Schema containing all enabled fields and attached metadata.

    Raises
    ------
    KeyError
        If any key in `schemaInfo` is missing its PyArrow type entry.
    """
    # Build list of fields with valid PyArrow types
    try:
        fields = [pa.field(name, schemaInfo[name][1]) for name in schemaInfo if schemaInfo[name][1] is not None and name in columns]
    except KeyError as e:
        raise KeyError(f"Field '{e.args[0]}' missing PyArrow type in schemaInfo")

    schema = pa.schema(fields)

    # Attach metadata if provided
    if metadata is not None:
        schema = schema.with_metadata(metadata)

    return schema


def convert_df_with_schema(df: pd.DataFrame, schemaInfo: Dict[str, Tuple] = SAMPIC_Schema_Info) -> pd.DataFrame:
    """
    Cast DataFrame columns to types defined in the schema information.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame whose columns should be converted in-place.
    schemaInfo : dict of str -> tuple
        Mapping of column names to schema tuples of the form:
        (pandas_dtype, pyarrow_type, numpy_dtype, [optional array size]).
        Only the pandas_dtype at index 0 is used for conversion when not None.

    Returns
    -------
    pandas.DataFrame
        The same DataFrame with its columns cast to the specified pandas dtypes.

    Notes
    -----
    - Columns not present in schemaInfo or with None pandas dtype are left unchanged.
    - Conversion is done in-place; the returned DataFrame is the same object.
    """
    for col in df.columns:
        dtype_info = schemaInfo.get(col)
        if dtype_info is None:
            continue
        pandas_dtype = dtype_info[0]
        if pandas_dtype:
            df[col] = df[col].astype(pandas_dtype)
    return df


def get_root_data_with_schema(df: pd.DataFrame, schemaInfo: Dict[str, Tuple] = SAMPIC_Schema_Info) -> Dict[str, np.ndarray]:
    """
    Prepare a dictionary of numpy arrays from a DataFrame for ROOT writing.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing fields defined in the schema.
    schemaInfo : dict of str -> tuple
        Mapping of column names to schema tuples of the form:
        (pandas_dtype, pyarrow_type, numpy_dtype, [optional array size]).
        The numpy_dtype at index 2 is used for array construction when not None.

    Returns
    -------
    dict
        Mapping of column names to numpy.ndarray with appropriate dtype.

    Raises
    ------
    ValueError
        If conversion of any column fails due to incompatible data or dtype.

    Notes
    -----
    - Only columns with a non-None numpy dtype in schemaInfo are included.
    - Any ValueError during conversion prints the offending column names for debugging.
    """
    ret_val: Dict[str, np.ndarray] = {}
    for col, info in schemaInfo.items():
        if col not in df.columns:
            continue
        numpy_dtype = None
        if len(info) > 2:
            numpy_dtype = info[2]
        if numpy_dtype is None:
            continue
        try:
            # Convert column to numpy array of target dtype
            # ret_val[col] = np.array(df[col].values, dtype=numpy_dtype)
            if len(info) > 3:
                ret_val[col] = np.array(df[col].tolist(), dtype=numpy_dtype)
            else:
                ret_val[col] = np.array(df[col], dtype=numpy_dtype)
        except Exception as e:
            # Debugging output
            print(f"Conversion error for column '{col}': dtype {numpy_dtype}")
            raise e
    return ret_val


def build_empty_root_data_with_schema(schemaInfo: Dict[str, Tuple] = SAMPIC_Schema_Info) -> Dict[str, np.ndarray]:
    """
    Construct an empty data dictionary for ROOT branches based on schema.

    Parameters
    ----------
    schemaInfo : dict of str -> tuple
        Mapping of column names to schema tuples of the form:
        (pandas_dtype, pyarrow_type, numpy_dtype, [optional array size]).
        The numpy_dtype at index 2 and optional array size at index 3 are used.

    Returns
    -------
    dict
        Mapping of column names to empty numpy.ndarray with correct shape and dtype.

    Notes
    -----
    - Scalar fields produce 1D arrays of length zero.
    - Fixed-size array fields (length provided in schemaInfo[3]) produce 2D arrays
      with shape (0, size).
    """
    ret_val: Dict[str, np.ndarray] = {}
    for col, info in schemaInfo.items():
        numpy_dtype = None
        arr_size = None
        if len(info) > 2:
            numpy_dtype = info[2]
        if len(info) > 3:
            arr_size = info[3]
        if numpy_dtype is None:
            continue
        if arr_size is not None:
            # Fixed-length vector field
            ret_val[col] = np.empty((0, arr_size), dtype=numpy_dtype)
        else:
            ret_val[col] = np.empty(0, dtype=numpy_dtype)
    return ret_val


def prepare_header_metadata_in_bytes(metadata: Dict[str, Any]) -> Dict[bytes, bytes]:
    """
    Convert a metadata dictionary of Python values to a bytes-to-bytes mapping suitable for Arrow.

    This utility encodes string, datetime, integer, and boolean metadata values
    into raw bytes for storage in Feather or Parquet file metadata.  Unsupported
    keys are skipped with a warning.

    Parameters
    ----------
    metadata : dict of str -> object
        Mapping of metadata keys to Python values.  Supported keys and types:

        - Text fields (ASCII strings):
          'sampiclyser_version', 'software_version', 'sampic_mezzanine_board_version',
          'ctrl_fpga_firmware_version', 'sampling_frequency',
          'hit_number_format', 'unix_time_format', 'data_format',
          'trigger_position_format', 'data_samples_format'
        - Timestamp field (datetime.datetime):
          'timestamp'
        - Integer fields (int):
          'num_channels', 'enabled_channels_mask', 'data_in_file_type'
        - Boolean flags (bool):
          'reduced_data_type', 'without_waveform', 'tdc_like_files',
          'inl_correction', 'adc_correction', 'compact_binary_data'

    Returns
    -------
    dict of bytes -> bytes
        Mapping of ASCII-encoded keys to packed byte values:
        - String values → ASCII-encoded bytes
        - Timestamp → little-endian IEEE-754 double of POSIX seconds
        - Integers → little-endian uint32
        - Booleans → single byte 0x01 (True) or 0x00 (False)

    Raises
    ------
    ValueError
        If a value has an unexpected type for a supported key.

    Notes
    -----
    - Keys not recognized in the supported list are skipped with a printed warning.
    - Use this mapping as the `schema.metadata` for PyArrow Schema or
      as the `metadata` argument in Feather/Parquet writers.
    """
    retVal: Dict[bytes, bytes] = {}

    for key, val in metadata.items():
        new_key = key.encode('ascii')

        # String fields
        if key in [
            'sampiclyser_version',
            'software_version',
            'sampic_mezzanine_board_version',
            'ctrl_fpga_firmware_version',
            'sampling_frequency',
            'hit_number_format',
            'unix_time_format',
            'data_format',
            'trigger_position_format',
            'data_samples_format',
        ]:
            if not isinstance(val, str):
                raise ValueError(f"Expected str for metadata '{key}', got {type(val)}")
            new_val = val.encode('ascii')

        # Timestamp field
        elif key == 'timestamp':
            if not isinstance(val, datetime):
                raise ValueError(f"Expected datetime for metadata 'timestamp', got {type(val)}")
            # POSIX timestamp in seconds
            new_val = struct.pack('<d', val.timestamp())

        # Integer fields
        elif key in ['num_channels', 'enabled_channels_mask', 'data_in_file_type']:
            if not isinstance(val, int):
                raise ValueError(f"Expected int for metadata '{key}', got {type(val)}")
            new_val = struct.pack('<I', val)

        # Boolean flags
        elif key in [
            'reduced_data_type',
            'without_waveform',
            'tdc_like_files',
            'inl_correction',
            'adc_correction',
            'compact_binary_data',
        ]:
            if not isinstance(val, bool):
                raise ValueError(f"Expected bool for metadata '{key}', got {type(val)}")
            new_val = b'\x01' if val else b'\x00'

        else:
            # Skip unknown keys
            print(f"Warning: skipping unknown metadata field '{key}'")
            continue

        retVal[new_key] = new_val

    return retVal


@dataclass
class SampicHeader:
    """
    Parsed header metadata from a SAMPIC file.

    Attributes
    ----------
    sampiclyser_version : str
        Version of the sampiclyser used to process the data.
    software_version : str
        Version of the SAMPIC DAQ software.
    timestamp : datetime.datetime
        Run start timestamp as a Python datetime.
    sampic_mezzanine_board_version : str
        Version identifier of the mezzanine board.
    num_channels : int
        Total number of channels in the system for this run.
    ctrl_fpga_firmware_version : str
        Version of the control FPGA firmware.
    front_end_fpga_firmware_version : list of str
        Firmware versions for each front-end FPGA.
    front_end_fpga_baseline : list of float
        Baseline values for each front-end FPGA, affecting all associated ADC channels.
    sampling_frequency : str
        System data acquisition sampling frequency specification.
    enabled_channels_mask : int
        Bitmask indicating which channels were enabled.
    reduced_data_type : bool
        Whether reduced-data format was used.
    without_waveform : bool
        Whether waveform data were omitted.
    tdc_like_files : bool
        Whether files are in TDC-like format.
    compact_binary_data: bool
        Whether the files are in the new compact binary format. Default: False
    data_in_file_type: int
        Which binary format to use, if the data is in compact binary format. Default: 0
    hit_number_format : str
        Format string for hit numbering.
    unix_time_format : str
        Format string for Unix timestamps.
    data_format : str
        Format string for data values.
    trigger_position_format : str
        Format string for trigger-position values. Optional.
    data_samples_format : str
        Format string for the data-sample values.
    inl_correction : bool
        Whether INL correction was applied.
    adc_correction : bool
        Whether ADC correction was applied.
    extra : dict of str → str
        Any unrecognized header fields (key/value both decoded as ASCII).
    """

    from sampiclyser import __version__

    sampiclyser_version: str = __version__
    software_version: str = ""
    timestamp: datetime | None = field(default=None, compare=False)
    sampic_mezzanine_board_version: str = ""
    num_channels: int = 0
    ctrl_fpga_firmware_version: str = ""
    front_end_fpga_firmware_version: List[str] = field(default_factory=list)
    front_end_fpga_baseline: List[float] = field(default_factory=list)
    sampling_frequency: str = ""
    enabled_channels_mask: int = 0
    reduced_data_type: bool = False
    without_waveform: bool = False
    tdc_like_files: bool = True
    compact_binary_data: bool = False
    data_in_file_type: int = 0
    hit_number_format: str = ""
    unix_time_format: str = ""
    data_format: str = ""
    trigger_position_format: str | None = field(default=None, compare=False)
    data_samples_format: str = ""
    inl_correction: bool = False
    adc_correction: bool = False
    extra: dict[str, str] = field(default_factory=dict, compare=False)


class SAMPIC_Run_Decoder:
    """
    Decode and process a complete SAMPIC run.

    Provides a one-pass, memory-efficient workflow for:
      1. Reading raw SAMPIC binary files from a run directory.
      2. Extracting and decoding header metadata.
      3. Streaming hit records in fixed-size chunks.
      4. Writing decoded hits and metadata to Feather, Parquet, or ROOT formats.

    The class preserves metadata both as raw bytes (for Arrow/Parquet) and
    as native Python types (for ROOT), and supports arbitrarily large files
    without loading everything into memory.

    Attributes
    ----------
    run_base_path : pathlib.Path
        Path to the directory containing all binary files for one run.
    run_header : SampicHeader
        Parsed header metadata for the current file being processed.
    run_files : list[pathlib.Path]
        List of all SAMPIC binary files in `run_base_path`, in sort order.
    """

    front_end_fpga_re = re.compile(r"^FRONT-END FPGA INDEX: (\d+) FIRMWARE VERSION (.+) BASELINE VALUE: ([\d\.]+)")
    timestamp_re = re.compile(r"^UnixTime = (.+) date = (.+) time = (.+ms)")

    def __init__(
        self,
        run_dir_path: Path,
    ):
        """
        Initialize a SAMPIC run decoder.

        Parameters
        ----------
        run_dir_path : pathlib.Path
            Directory containing the SAMPIC binary files for a single run.

        Raises
        ------
        FileNotFoundError
            If `run_dir_path` does not exist or is not a directory.
        """
        self.run_base_path = run_dir_path

        all_files = natsorted(list(self.run_base_path.glob("*.bin*")))

        self.trigger_files = [f for f in all_files if "_trigger_data.bin" in f.name]
        self.run_files = [f for f in all_files if "_trigger_data.bin" not in f.name]

    @contextmanager
    def open_sampic_file_in_chunks_and_get_header(
        self,
        file_path: Path,
        extra_header_bytes: int,
        chunk_size: int = 64 * 1024,
        debug: bool = False,
    ) -> Generator[Tuple[bytes, Generator[bytes, None, None]], None, None]:
        """
        Memory-map a SAMPIC file, extract its header, and stream the remainder in chunks.

        This context manager opens `file_path` in read-only mode, mmaps the entire
        file, and locates the header boundary as the last '=' byte before the first
        `0x00`.  It returns the header (including `extra_header_bytes`) and a
        generator yielding the file body in `chunk_size`-byte blocks.  On exit,
        both the file and the mmap are cleanly closed.

        During this process, `self.current_filesize` is set to the size of the file.

        Parameters
        ----------
        file_path : pathlib.Path
            Path to the binary SAMPIC file to read.
        extra_header_bytes : int
            Number of bytes to include *after* the header delimiter (`=`) in the
            returned header.
        chunk_size : int, optional
            Size of each chunk (in bytes) produced by the body generator.
            Default is 64 KiB.
        debug : bool, optional
            If True, print debugging information.  Default is False.

        Yields
        ------
        header_bytes : bytes
            The raw header bytes, from the file start up through the computed end.
        body_gen : generator of bytes
            Generator yielding successive `chunk_size`-byte slices of the file body.

        Raises
        ------
        ValueError
            If the header delimiter cannot be located (i.e. no '=' before the
            first 0x00), indicating a malformed file.
        """
        f = file_path.open('rb')

        try:
            self.current_filesize = file_path.stat().st_size

            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            mm_size = mm.size()

            # 1) find first null
            first_null = mm.find(b'\x00')
            if first_null <= 0:  # Explicitly do not accept situation where the first null is the first byte in the file
                raise ValueError("No null byte (0x00) found in file")

            # 2) within header region, find last '='
            last_eq = mm.rfind(b'===\n', 0, first_null)
            if last_eq < 0:
                raise ValueError("No '=' found before first 0x00")

            # 3) compute header slice
            header_end = min(last_eq + 3 + extra_header_bytes, mm_size)
            header = mm[:header_end]

            if debug:
                print(file_path.name)
                print(header_end)

            # 4) define body generator
            def body_gen() -> Generator[bytes, None, None]:
                offset = header_end
                while offset < mm_size:
                    yield mm[offset : offset + chunk_size]
                    offset += chunk_size

            yield header, body_gen()

        finally:
            delattr(self, "current_filesize")
            mm.close()
            f.close()

    @staticmethod
    def _parse_header_field(  # noqa: max-complexity=20
        field: str,
        header: SampicHeader,
        keep_unparsed: bool = True,
    ) -> None:
        """
        Parse a single header field string and populate the corresponding attribute.

        This helper inspects a raw field fragment (text between “===” delimiters),
        extracts the key and value(s), converts them to the appropriate type, and
        assigns them on the supplied `SampicHeader` instance.  If the key is not
        one of the recognized header attributes and `keep_unparsed` is True, the
        raw field text is stored in `header.extra`.

        Parameters
        ----------
        field : str
            Raw header fragment, e.g. "param1: value1 part2 = 42".
        header : SampicHeader
            The dataclass instance to be populated in-place.
        keep_unparsed : bool, optional
            If True (default), any unrecognized field is appended to
            `header.extra` under its raw key; if False, unrecognized
            fields are ignored.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the field text cannot be split into a key and a value,
            or if a known key's value fails conversion (e.g. non-numeric
            text for an integer field).
        """
        if "==" in field:
            key = None
            sub_fields = [f.strip() for f in field.split("==") if f.strip()]
            for sub_field in sub_fields:
                SAMPIC_Run_Decoder._parse_header_field(sub_field, header, keep_unparsed=keep_unparsed)
        elif "  " in field and field[:4] != "Ch (":
            key = None
            sub_fields = [f.strip() for f in field.split("  ") if f.strip()]
            for sub_field in sub_fields:
                SAMPIC_Run_Decoder._parse_header_field(sub_field, header, keep_unparsed=keep_unparsed)
        elif "MEZZA_SAMPIC BOARD" == field[:18]:
            key = "MEZZA_SAMPIC BOARD"
            key_l = "sampic_mezzanine_board_version"
            val = field[19:]
        elif "NB OF CHANNELS IN SYSTEM" == field[:24]:
            key = "NB OF CHANNELS IN SYSTEM"
            key_l = "num_channels"
            val = int(field[25:])
        elif "CTRL FPGA FIRMWARE VERSION" == field[:26]:
            key = "CTRL FPGA FIRMWARE VERSION"
            key_l = "ctrl_fpga_firmware_version"
            val = field[27:]
        elif "SAMPLING FREQUENCY" == field[:18]:
            key = "SAMPLING FREQUENCY"
            key_l = "sampling_frequency"
            val = field[19:]
        elif "FRONT-END FPGA INDEX" == field[:20]:
            key = None
            match = SAMPIC_Run_Decoder.front_end_fpga_re.match(field)
            index = int(match.group(1))
            version = match.group(2)
            baseline = float(match.group(3))

            start_len = len(header.front_end_fpga_firmware_version)
            if start_len < index + 1:
                header.front_end_fpga_firmware_version = header.front_end_fpga_firmware_version + [None] * (index + 1 - start_len)
                header.front_end_fpga_baseline = header.front_end_fpga_baseline + [None] * (index + 1 - start_len)

            header.front_end_fpga_firmware_version[index] = version
            header.front_end_fpga_baseline[index] = baseline
        elif ":" in field:
            key, val = (p.strip() for p in field.split(":", 1))

            # Perform data conversion where needed
            key_l = None
            if "DATA FILE SAVED WITH SOFTWARE VERSION" == key:
                key_l = "software_version"
            elif "DATE OF RUN" == key:
                key = None

                match = SAMPIC_Run_Decoder.timestamp_re.match(val)

                dt_local = datetime.fromtimestamp(float(match.group(1)))

                key = "TIMESTAMP"
                key_l = "timestamp"
                val = dt_local
            elif "Enabled Channels Mask" == key:
                val = int(val, base=16)
            elif "REDUCED DATA TYPE" == key:
                if val == "NO":
                    val = False
                else:
                    val = True
            elif "WITHOUT WAVEFORM" == key:
                if val == "NO":
                    val = False
                else:
                    val = True
            elif "TDC-LIKE FILES" == key:
                if val == "YES":
                    val = True
                else:
                    val = False
            elif "COMPACT BINARY DATA" == key:
                if val == "YES":
                    val = True
                else:
                    val = False
            elif "DATA_IN_FILE_TYPE" == key:
                val = int(val, 10)
            elif "INL Correction" in key:
                index = key.find("INL Correction")
                SAMPIC_Run_Decoder._parse_header_field(key[: index - 5].strip(), header, keep_unparsed=keep_unparsed)
                key = key[index:]
                if val == "ON":
                    val = True
                else:
                    val = False
            elif "ADC Correction" == key:
                if val == "ON":
                    val = True
                else:
                    val = False

            if (key_l is None) and (key is not None):
                key_l = key.lower().replace(" ", "_").replace("-", "_")
        elif field[:4] == "Ch (":
            setattr(header, "data_format", field)
            key = None
        elif field[-1] == ']' or field[:11] == "DataSamples":
            index = field.find('[')
            final_index = field.find(']')

            key = field[:index]
            val = field[index + 1 : final_index]

            if "DataSamples" == key:
                key_l = "data_samples_format"
            elif "TriggerPosition" == key:
                key_l = "trigger_position_format"
            else:
                key_l = key
        elif field[-1] == ')':
            index = field.find('(')

            key = field[:index].strip()
            val = field[index + 1 : -1].strip()

            if "HIT number" == key:
                key_l = "hit_number_format"
            elif "UnixTime" == key:
                key_l = "unix_time_format"
            else:
                key_l = key
        else:
            # fallback: dump everything into extra with a generic key
            key = None
            if keep_unparsed:
                header.extra[f"unparsed_{len(header.extra)}"] = field

        if key is not None:
            if hasattr(header, key_l):
                setattr(header, key_l, val)
            else:
                header.extra[key] = val

    def decode_sampic_header(
        self,
        header_bytes: bytes,
        keep_unparsed: bool = True,
    ) -> SampicHeader:
        """
        Parse raw header bytes into a SampicHeader instance.

        The header consists of one or more lines; each line starts and ends
        with "===" and contains fields separated by "===".  Field syntax may
        vary (e.g. "key: value", "key value", or composite "part1 = x part2 = y").

        Parameters
        ----------
        header_bytes : bytes
            Raw bytes of the header section, from file start up to the header end
            (inclusive of delimiters and any extra bytes).
        keep_unparsed : bool, optional
            If True (default), any fields that are not recognized are stored in
            the `extra` dict of the returned SampicHeader; if False, they are discarded.

        Returns
        -------
        SampicHeader
            A dataclass containing all parsed header values and optionally any
            unrecognized fields in its `extra` attribute.

        Raises
        ------
        ValueError
            If the header_bytes cannot be decoded into valid text, or if required
            header fields are missing or malformed.

        Notes
        -----
        This method:
          1. Splits `header_bytes` on lines beginning/ending with "===".
          2. For each field fragment, calls `_parse_header_field`.
          3. Collects any unparsed text in `SampicHeader.extra`.
        """
        # Convert to text
        text = header_bytes.decode('utf-8', errors='replace')

        # Remove leading/trailing markers and split lines
        lines = [ln.strip()[3:-3].strip() for ln in text.splitlines() if ln.strip().startswith("===") and ln.strip().endswith("===")]

        # Start with defaults or placeholders
        header = SampicHeader()

        for line in lines:
            # split into fields by the === separator
            fields = [f.strip() for f in line.split("===") if f.strip()]
            for fld in fields:
                self._parse_header_field(fld, header, keep_unparsed=keep_unparsed)

        return header

    def parse_hit_records(  # noqa: max-complexity=26
        self,
        limit_hits: int = 0,
        extra_header_bytes: int = 1,
        chunk_size: int = 64 * 1024,
        coarse_freq_hz: int = 100_000_000,  # new
        coarse_bit_depth: int = 40,  # new
        timestamp_window_size: int = 64,  # new
        max_out_of_time_s: float = 10,  # new
        debug=False,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream and decode hit records from all files in the run.

        This generator method opens each SAMPIC binary file in turn, extracts
        its header (via `open_sampic_file_in_chunks_and_get_header`), checks
        for header consistency across files, then streams the body in fixed-size
        chunks, parsing out complete hit records until either the file ends or
        `limit_hits` is reached.

        Parameters
        ----------
        limit_hits : int, optional
            Maximum number of hit records to yield across all files.
            A value of 0 (default) means no limit (process all hits).
        extra_header_bytes : int, optional
            Number of bytes to include _after_ the header delimiter when
            extracting the header (default is 1 to include the newline).
        chunk_size : int, optional
            Size in bytes of each data chunk read from the body (default is
            64 KiB). Larger chunks may be more efficient but use more memory.

        Yields
        ------
        record : dict
            A mapping from field names (str) to parsed values (int, float,
            bool, list, etc.) for each hit record.

        Raises
        ------
        ValueError
            If header parsing fails or a file's header does not match the
            previously parsed header (mismatched run files).

        Notes
        -----
        - Uses a rolling buffer to accumulate bytes from the stream until a
          full record can be parsed by `try_parse_record`.
        - After parsing each record, advances the buffer and continues until
          all records are yielded or `limit_hits` is reached.
        """
        mismatched_header_errors = []

        buffer = bytearray()
        hits = 0

        def generate_field_specs():
            # Pre-Compile Struct Formats
            # See https://docs.python.org/3/library/struct.html
            s_u8 = Struct('B')
            s_i16 = Struct('<h')
            s_ui16 = Struct('<H')
            s_i32 = Struct('<i')
            # s_i64 = Struct('<q')
            s_ui64 = Struct('<Q')
            s_f32 = Struct('<f')
            s_f64 = Struct('<d')

            def get_field_parser(field_format: str):
                if field_format == "uchar":
                    return 1, lambda v, o: s_u8.unpack_from(v, o)[0], "S"
                elif field_format == "signed short":
                    return 2, lambda v, o: s_i16.unpack_from(v, o)[0], "S"
                elif field_format == "ushort":
                    return 2, lambda v, o: s_ui16.unpack_from(v, o)[0], "S"
                elif field_format == "int":
                    return 4, lambda v, o: s_i32.unpack_from(v, o)[0], "S"
                elif field_format == "u_int64":
                    return 8, lambda v, o: s_ui64.unpack_from(v, o)[0], "S"
                elif field_format == "float":
                    return 4, lambda v, o: s_f32.unpack_from(v, o)[0], "S"
                elif field_format == "double":
                    return 8, lambda v, o: s_f64.unpack_from(v, o)[0], "S"
                elif " " in field_format:
                    if field_format[:8] == "array of":
                        field_bytes, field_parser, _ = get_field_parser(field_format[9:])
                        return field_bytes, field_parser, "A[DataSize]"
                    elif field_format[0] == "(":
                        index = field_format.find(')')
                        return get_field_parser(field_format[index + 1 :].strip())
                    elif field_format.split(" ")[0] == "in":
                        field_bytes, field_parser, _ = get_field_parser(self.run_header.data_samples_format)
                        return field_bytes, field_parser, "S"
                    else:
                        try:
                            return get_field_parser(field_format.split(" ")[0])
                        except RuntimeError:
                            raise RuntimeError(f"Unknown field format: {field_format}")
                else:
                    raise RuntimeError(f"Unknown field format: {field_format}")

            field_specs = []

            def append_field(field_name: str, field_format: str):
                if debug:
                    print(f"Field: {field_name}; Format: {field_format}")
                field_bytes, field_parser, field_type = get_field_parser(field_format)
                field_specs.append((field_name, field_bytes, field_parser, field_type))

            if not self.run_header.compact_binary_data:
                # Fixed information in header (so far guaranteed to always exist)
                # Hit Number is first field
                append_field("HitNumber", self.run_header.hit_number_format)

                # Unix timestamp is next
                append_field("UnixTime", self.run_header.unix_time_format)

                # Variable information (extracted from header format for the hits)
                pattern = r'\(?\s*([^\s()]+)\s*\(((?:[^()]+|\([^)]*\))*)\)'
                for match in re.finditer(pattern, self.run_header.data_format):
                    field_name = match.group(1)
                    field_format = match.group(2)

                    append_field(field_name, field_format)

                # Array information of the hits
                if self.run_header.trigger_position_format is not None:
                    append_field("TriggerPosition", self.run_header.trigger_position_format)

                append_field("DataSample", self.run_header.data_samples_format)
            elif self.run_header.data_in_file_type == 0:
                # Structure from SAMPIC data format from 18/04/2023

                # Hit Number is first field
                append_field("HitNumber", "int")

                # Channel
                append_field("Channel", "uchar")

                # TimeStamp
                append_field("FirstSampleTimeStamp", "double")

                # RawTOTValue
                append_field("RawTOTValue", "ushort")

                # TOTValue
                append_field("TOTValue", "float")

                # FirstCellIndex
                append_field("FirstCellIndex", "uchar")

                # DataSize
                append_field("DataSize", "uchar")

                # DataSamples
                append_field("DataSample", "array of signed short")
            elif self.run_header.data_in_file_type == 1:
                # Structure from SAMPIC data format from 18/04/2023

                # Hit Number is first field
                append_field("HitNumber", "int")

                # Channel
                append_field("Channel", "uchar")

                # TimeStamp
                append_field("FirstSampleTimeStamp", "double")

                # RawTOTValue
                append_field("RawTOTValue", "ushort")

                # TOTValue
                append_field("TOTValue", "float")

                # Time
                append_field("Time", "float")

                # Baseline
                append_field("Baseline", "float")

                # Amplitude
                append_field("Amplitude", "float")

                # FirstCellIndex
                append_field("FirstCellIndex", "uchar")

                # DataSize
                append_field("DataSize", "uchar")

                # DataSamples
                append_field("DataSample", "array of signed short")
            elif self.run_header.data_in_file_type == 2:
                # Structure from SAMPIC data format from 18/04/2023

                # Channel
                append_field("Channel", "uchar")

                # TimeStamp
                append_field("FirstSampleTimeStamp", "double")

                # RawTOTValue
                append_field("RawTOTValue", "ushort")

                # TOTValue
                append_field("TOTValue", "float")

                # Time
                append_field("Time", "double")

                # Baseline
                append_field("Baseline", "float")

                # Amplitude
                append_field("Amplitude", "float")
            elif self.run_header.data_in_file_type == 3:
                # Structure from SAMPIC data format from 18/04/2023

                # Channel
                append_field("Channel", "uchar")

                # Time
                append_field("Time", "double")

                # TOTValue
                append_field("TOTValue", "float")

                # RawTOTValue
                append_field("RawTOTValue", "ushort")

            if debug:
                print("Field Specs:")
                for val in field_specs:
                    print(f"  - {val}")

            return field_specs

        timestamp_fifo = deque(maxlen=timestamp_window_size)
        self.printed_double_time_resolution_warning = False
        run_start = None
        max_time_counter_ps = calculate_wrap_offset_ps(coarse_bit_depth, coarse_freq_hz)

        # Helper to try parsing one record from the buffer
        def try_parse_record(field_specs) -> Dict[str, Any] | None:
            view = memoryview(buffer)

            # First ensure we have at least the fixed portion
            fixed_len = sum(n for _, n, _, t in field_specs if t == "S")
            if len(buffer) < fixed_len:
                return None  # need more data

            # Parse fixed fields to extract the fixed portion, counts should be in this portion so we can parse the rest of the data structure
            record: Dict[str, Any] = {
                # Coordination
                "HitNumber": None,
                # From SAMPIC
                "Channel": None,  # Always present
                "FirstSampleTime_in_ps": None,
                "FirstSampleTime_in_ps_fine": None,
                "RawTOTValue": None,  # Always present
                "TOTValue": None,  # Always present
                # From SAMPIC: Samples / Waveform
                "DataSize": None,
                "DataSample": None,
                # Measurements
                "Time": None,
                "Baseline": None,
                "RawPeak": None,
                "Amplitude": None,
                # Other
                "FirstCellIndex": None,
                "NumTriggerSamples": None,
            }
            offset = 0
            for name, nbytes, conv, field_type in field_specs:
                if field_type != "S":
                    break  # Stop once we find the first non scalar/single type field
                # record[name] = conv(view[offset : offset + nbytes])
                record[name] = conv(view, offset)
                offset += nbytes

            # Compute total required length (fixed + arrays)
            total_len = 0
            for name, nbytes, _, field_type in field_specs:
                multiplier = 1
                if field_type != "S":  # For non-scalar field types:
                    if field_type[0] == 'A':  # For Vector and Array field types
                        if field_type[1] != '[' or field_type[-1] != ']':
                            raise RuntimeError(
                                f"Malformed field type for {name}, please double check configuration is correct: {field_type}"
                            )
                        count_name = field_type[2:-1]
                        if count_name not in record:
                            raise RuntimeError(
                                f"Could not find the count field ({count_name}), either the name is wrong or the field is not in the fixed portion of the record"
                            )
                        multiplier = record[count_name]
                    else:
                        raise RuntimeError(
                            "Unknown field type defined, unable to parse data, so we are aborting since we can not guarantee the data is correctly interpreted"
                        )
                total_len += nbytes * multiplier
            if len(buffer) < total_len:
                return None  # wait for more data

            # Parse remaining data, including arrays
            offset = 0
            for name, nbytes, conv, field_type in field_specs:
                multiplier = 1
                if field_type != "S":  # For non-scalar field types:
                    if field_type[0] == 'A':  # For Vector and Array field types
                        # Don't need to perform check below because they were performed above
                        # if field_type[1] != '[' or field_type[-1] != ']':
                        #    raise RuntimeError(f"Malformed field type for {name}, please double check configuration is correct: {field_type}")
                        count_name = field_type[2:-1]
                        # if count_name not in record:
                        #    raise RuntimeError(f"Could not find the count field ({count_name}), either the name is wrong or the field is not in the fixed portion of the record")
                        multiplier = record[count_name]
                        array = []
                        for i in range(multiplier):
                            start = offset + i * nbytes
                            # end   = start + nbytes

                            # array.append(conv(view[start : end]))
                            array.append(conv(view, start))
                        record[name] = array
                    # else:
                    #    raise RuntimeError("Unknown field type defined, unable to parse data, so we are aborting since we can not guarantee the data is correctly interpreted")
                    else:
                        pass
                else:
                    if name not in record:
                        # record[name] = conv(view[offset : offset + nbytes])
                        record[name] = conv(view, offset)
                offset += nbytes * multiplier

            # Normalize data here
            if "Ch" in record:
                record['Channel'] = record.pop("Ch")

            if not self.run_header.compact_binary_data:
                if "UnixTime" not in record:
                    raise RuntimeError("Can't find the UnixTime when it should exist... perhaps the data format changed... again :(")
                timestamp = record["UnixTime"]
                if "OrderedCell0Time" in record:
                    cell0Time = record["OrderedCell0Time"]
                elif "Cell0Time" in record:
                    cell0Time = record["Cell0Time"]
                else:
                    raise RuntimeError("Can't find the Cell0Time for ordering hits... perhaps the data format changed... again :( ")

                if not self.printed_double_time_resolution_warning and cell0Time >= MAX_1PS_RESOLUTION_NS:
                    print(
                        colored("Warning:", "yellow"),
                        f"File contains hits where hit time (OrderedCell0Time or cell0Time) exceed {MAX_1PS_RESOLUTION_NS} ns. These hits will have inherent time resolution worse than 1 ps.",
                    )
                    self.printed_double_time_resolution_warning = True

                # fetch unix time and compare against timestamp. if timestamp is significantly smaller, then a loop occured, so add a multiple of 3.05 to get good time

                # Convert cell0Time into seconds and compare against the elapsed time according to the UnixTimestamp
                delta = (timestamp - run_start) - cell0Time / 10**9
                wraps = floor((delta + max_out_of_time_s) / (max_time_counter_ps / 10**12))

                hit_time_ps, hit_time_remainder_ps = convert_daq_ns_double_to_ps_int(cell0Time, wraps, max_time_counter_ps)

                new_delta = (timestamp - run_start) - hit_time_ps / 10**12
                if new_delta > max_out_of_time_s:
                    print(
                        colored("Warning:", "yellow"),
                        f"After unwrapping, we still observe a large time difference (Delta={new_delta}), this could indicate some unexpected problem.",
                    )

                record["FirstSampleTime_in_ps"] = hit_time_ps
                record["FirstSampleTime_in_ps_fine"] = hit_time_remainder_ps
            elif self.run_header.data_in_file_type < 3:
                if "FirstSampleTimeStamp" in record:
                    cell0Time = record["FirstSampleTimeStamp"]
                else:
                    raise RuntimeError(
                        "Can't find the FirstSampleTimeStamp for ordering hits... perhaps the data format changed... again :( "
                    )

                if not self.printed_double_time_resolution_warning and cell0Time >= MAX_1PS_RESOLUTION_NS:
                    print(
                        colored("Warning:", "yellow"),
                        f"File contains hits where hit time (FirstSampleTimeStamp) exceed {MAX_1PS_RESOLUTION_NS} ns. These hits will have inherent time resolution worse than 1 ps.",
                    )
                    self.printed_double_time_resolution_warning = True

                # Here we use a different algorithm where we need to identify when the counter goes wraps,
                # but we do not have a UnixTimestamp to help guide us.
                # I decided to approach this by having a FIFO of the last N unwrapped hit times. Then I just need
                # to find the number of wraps that brings the current hit closest to the mean of the FIFO. Should
                # be robust against all effects smaller than half the FIFO size

                if len(timestamp_fifo) == 0:
                    wraps = 0
                else:
                    expected_absolute_time_ps = statistics.median(timestamp_fifo)

                    # Get only the ps part of cell0Time, without any wraps, to compare against expected
                    intermediate_time_ps, _ = convert_daq_ns_double_to_ps_int(cell0Time, 0, max_time_counter_ps)

                    wraps = round((expected_absolute_time_ps - intermediate_time_ps) / (max_time_counter_ps))

                hit_time_ps, hit_time_remainder_ps = convert_daq_ns_double_to_ps_int(cell0Time, wraps, max_time_counter_ps)

                record["FirstSampleTime_in_ps"] = hit_time_ps
                record["FirstSampleTime_in_ps_fine"] = hit_time_remainder_ps

                timestamp_fifo.append(hit_time_ps)

            if not self.run_header.compact_binary_data and "CellInfo" in record:
                record["FirstCellIndex"] = 64 - record.pop("CellInfo")

            if not self.run_header.compact_binary_data and "TriggerPosition" in record:
                triggers: List = record.pop("TriggerPosition")
                record["NumTriggerSamples"] = triggers.count(1)

            # Add here to reorder the samples inside the hit if needed
            # if not self.run_header.compact_binary_data:
            #    pass
            # else:
            #    pass

            if self.run_header.compact_binary_data and self.run_header.adc_correction:
                record["DataSample"] = [sample / 10000.0 for sample in record["DataSample"]]

            # Remove empty stuff
            keys = list(record.keys())
            for key in keys:
                if record[key] is None:
                    record.pop(key)

            # consume bytes from buffer
            del view
            del buffer[:total_len]
            return record

        first_header = None
        return_now = False
        for file in self.run_files:
            with self.open_sampic_file_in_chunks_and_get_header(file, extra_header_bytes, chunk_size) as (raw_header, body_gen):
                header = self.decode_sampic_header(raw_header, keep_unparsed=True)
                if first_header is not None:
                    if not (header == first_header):
                        mismatched_header_errors.append(file)
                else:
                    first_header = header
                    self.run_header = header
                    field_specs = generate_field_specs()
                    run_start = self.run_header.timestamp.timestamp()

                    # Print out warnings about assumptions for time reconstruction
                    if self.run_header.compact_binary_data and self.run_header.data_in_file_type < 3:
                        print(
                            colored("Warning:", "yellow"),
                            "The RAW data file was taken in a compact mode and with hit time information (when each hit happened). In this mode the time counter appears to wrap at the 3.05 hour mark (40 bits at 100Mhz maybe?), any run longer than 3.05h will not be able to uniquely determine when a hit happened, the hit ordering helps but it is impossible to know with certainty if only 1 or more wraps happened between any 2 hits. Execution will proceed assuming the system was configured such that at least one channel had a few hits within the 3.05h period, thus each decrease in the timestamp is associated with one wrap of the counter (modulo the out of order hits).",
                        )
                    if self.run_header.compact_binary_data and self.run_header.data_in_file_type == 3:
                        print(
                            colored("Warning:", "yellow"),
                            "The RAW data file was taken in a compact mode and without hit time information (when each hit happened). In this mode, a lot of the SAMPIClyser analysis can not be performed. SAMPIClyser will still convert the file and make the information that is present available in the unified interface.",
                        )
                    if self.run_header.compact_binary_data and self.run_header.data_in_file_type > 3:
                        print(
                            colored("Error:", "red"),
                            f"Unknown compact data format (type: {self.run_header.data_in_file_type}), this should not exist according to the documentation the binary converter was based on. Binary conversion will likely fail and give weird errors.",
                        )
                    if not self.run_header.compact_binary_data:  # Is the comment below still true if not compact and reduced is selected?
                        print(
                            colored("Warning:", "yellow"),
                            "The RAW data file was taken with hit time information (when each hit happened). In this mode the time counter appears to wrap at the 3.05 hour mark (40 bits at 100Mhz maybe?), any run longer than 3.05h will not be able to uniquely determine when a hit happened, the hit ordering helps but it is impossible to know with certainty if only 1 or more wraps happened between any 2 hits. This mode also has a Unix Timestamp of when the hit data reached the computer, this helps restore time ordering. Execution will proceed assuming this information is available and will use it to restore correct timing.",
                        )

                # Stream through chunks
                for chunk in body_gen:
                    buffer.extend(chunk)

                    # parse as many complete records as we can
                    while True:
                        if limit_hits > 0 and hits >= limit_hits:
                            return_now = True
                            break
                        rec = try_parse_record(field_specs)
                        if rec is None:
                            break
                        hits += 1
                        yield rec

                    if return_now:
                        break
            if return_now:
                break

        # Cleanup
        if len(mismatched_header_errors) > 0:
            print(
                colored("Warning:", "yellow"),
                f"Found mismatches in the headers of the files that make this run. The list of mismatched files is: {mismatched_header_errors}",
            )

        return

    def prepare_header_metadata(self) -> Dict[bytes, bytes]:
        """
        Pack run-header attributes into raw byte metadata for columnar files.

        Generates a mapping of metadata keys to byte-encoded values suitable
        for Arrow/Parquet file schemas, preserving binary precision and type.

        Returns
        -------
        metadata : dict of bytes → bytes
            Byte-to-byte mapping where:

            - Text fields (e.g. software_version) are ASCII-encoded.
            - `timestamp` is a little-endian 8-byte float (`struct.pack('<d', ...)`).
            - `num_channels` and `enabled_channels_mask` are little-endian
              4-byte unsigned ints (`struct.pack('<I', ...)`).
            - Boolean flags (`reduced_data_type`, `without_waveform`, etc.)
              are stored as a single byte: `b'\x00'` for False, `b'\x01'` for True.

        Notes
        -----
        Keys are raw byte strings (e.g. `b'software_version'`), matching the
        Arrow metadata API expectations. This preserves full fidelity for
        programmatic reloading via `decode_byte_metadata`.
        """
        retVal: Dict[bytes, bytes] = {
            b'sampiclyser_version': self.run_header.sampiclyser_version.encode('ascii'),
            b'software_version': self.run_header.software_version.encode('ascii'),
            b'timestamp': struct.pack('<d', self.run_header.timestamp.timestamp()),
            b'sampic_mezzanine_board_version': self.run_header.sampic_mezzanine_board_version.encode('ascii'),
            b'num_channels': struct.pack('<I', self.run_header.num_channels),
            b'ctrl_fpga_firmware_version': self.run_header.ctrl_fpga_firmware_version.encode('ascii'),
            # front_end_fpga_firmware_version: List[str] = field(default_factory=list)
            # front_end_fpga_baseline: List[float] = field(default_factory=list)
            b'sampling_frequency': self.run_header.sampling_frequency.encode('ascii'),
            b'enabled_channels_mask': struct.pack('<I', self.run_header.enabled_channels_mask),
            b'reduced_data_type': b'\x01' if self.run_header.reduced_data_type else b'\x00',
            b'without_waveform': b'\x01' if self.run_header.without_waveform else b'\x00',
            b'tdc_like_files': b'\x01' if self.run_header.tdc_like_files else b'\x00',
            b'compact_binary_data': b'\x01' if self.run_header.compact_binary_data else b'\x00',
            b'data_in_file_type': struct.pack('<I', self.run_header.data_in_file_type),
            b'hit_number_format': self.run_header.hit_number_format.encode('ascii'),
            b'unix_time_format': self.run_header.unix_time_format.encode('ascii'),
            b'data_format': self.run_header.data_format.encode('ascii'),
            # b'trigger_position_format': self.run_header.trigger_position_format.encode('ascii'),
            b'data_samples_format': self.run_header.data_samples_format.encode('ascii'),
            b'inl_correction': b'\x01' if self.run_header.inl_correction else b'\x00',
            b'adc_correction': b'\x01' if self.run_header.adc_correction else b'\x00',
        }

        if self.run_header.trigger_position_format is not None:
            retVal[b'trigger_position_format'] = self.run_header.trigger_position_format.encode('ascii')

        return retVal

    def prepare_root_header_metadata(self) -> Dict[str, object]:
        """
        Build a Python-native metadata dict for ROOT TTree output.

        Collects all run-header fields into native Python types so they can be
        written directly as branches in a ROOT metadata TTree.

        Returns
        -------
        metadata : dict of str → object
            Dictionary mapping metadata keys to Python values, including:

            - `sampiclyser_version` : str
            - `software_version` : str
            - `timestamp` : datetime.datetime
            - `sampic_mezzanine_board_version` : str
            - `num_channels` : int
            - `ctrl_fpga_firmware_version` : str
            - `sampling_frequency` : str
            - `enabled_channels_mask` : int
            - `reduced_data_type` : bool
            - `without_waveform` : bool
            - `tdc_like_files` : bool
            - `compact_binary_data` : bool
            - `data_in_file_type` : int
            - `hit_number_format` : str
            - `unix_time_format` : str
            - `data_format` : str
            - `trigger_position_format` : str
            - `data_samples_format` : str
            - `inl_correction` : bool
            - `adc_correction` : bool

        Notes
        -----
        All values are in their natural Python form (no byte-packing), ready
        for conversion to Awkward or NumPy arrays when writing via uproot.
        """
        retVal: Dict[str, object] = {
            'sampiclyser_version': self.run_header.sampiclyser_version,
            'software_version': self.run_header.software_version,
            'timestamp': self.run_header.timestamp,
            'sampic_mezzanine_board_version': self.run_header.sampic_mezzanine_board_version,
            'num_channels': self.run_header.num_channels,
            'ctrl_fpga_firmware_version': self.run_header.ctrl_fpga_firmware_version,
            # front_end_fpga_firmware_version: List[str] = field(default_factory=list)
            # front_end_fpga_baseline: List[float] = field(default_factory=list)
            'sampling_frequency': self.run_header.sampling_frequency,
            'enabled_channels_mask': self.run_header.enabled_channels_mask,
            'reduced_data_type': self.run_header.reduced_data_type,
            'without_waveform': self.run_header.without_waveform,
            'tdc_like_files': self.run_header.tdc_like_files,
            'compact_binary_data': self.run_header.compact_binary_data,
            'data_in_file_type': self.run_header.data_in_file_type,
            'hit_number_format': self.run_header.hit_number_format,
            'unix_time_format': self.run_header.unix_time_format,
            'data_format': self.run_header.data_format,
            # 'trigger_position_format': self.run_header.trigger_position_format,
            'data_samples_format': self.run_header.data_samples_format,
            'inl_correction': self.run_header.inl_correction,
            'adc_correction': self.run_header.adc_correction,
        }

        if self.run_header.trigger_position_format is not None:
            retVal['trigger_position_format'] = self.run_header.trigger_position_format

        return retVal

    def write_root_header(self, froot: uproot.WritableDirectory) -> None:
        """
        Embed run-header metadata into a ROOT file as a metadata TTree.

        Converts the dict returned by `prepare_root_header_metadata` into
        Awkward arrays of strings and writes them as two branches
        ('key' and 'value') in a TTree named 'metadata'. Existing
        metadata trees of the same name are overwritten.

        Parameters
        ----------
        froot : uproot.WritableDirectory
            An open ROOT file handle (from `uproot.recreate` or `uproot.update`)
            into which the metadata TTree will be written.

        Returns
        -------
        None

        Notes
        -----
        - Keys and values are both stored as variable-length strings using
          Awkward Arrays (`ak.from_iter`).
        - The resulting TTree will have two string branches:
            - `key`   : metadata field names
            - `value` : metadata field values (all converted to str)
        - If a 'metadata' TTree already exists, it is replaced.
        """
        metadata = self.prepare_root_header_metadata()

        # Convert to Awkward Arrays of strings for variable-length support
        keys = ak.from_iter(list(metadata.keys()), highlevel=True)
        vals = ak.from_iter([str(v) for v in metadata.values()], highlevel=True)

        # Assign the new metadata TTree
        froot['metadata'] = {
            'key': keys,
            'value': vals,
        }

    def decode_data(  # noqa: max-complexity=24
        self,
        limit_hits: int = 0,
        feather_path: Optional[Path] = None,
        parquet_path: Optional[Path] = None,
        root_path: Optional[Path] = None,
        root_tree: str = "sampic_hits",
        extra_header_bytes: int = 1,
        chunk_size: int = 64 * 1024,
        batch_size: int = 100_000,
        debug: bool = False,
    ) -> None:
        """
        Decode hit records from SAMPIC run files and export to Feather, Parquet, and/or ROOT.

        This method streams parsed hit-record dictionaries (via
        `parse_hit_records`), accumulates them in batches to build a
        pandas DataFrame, and then writes each batch to the specified
        output formats.  It never holds all records in memory at once.

        Parameters
        ----------
        limit_hits : int, optional
            Maximum number of hit records to process across all run files.
            A value of 0 (default) means “no limit” (process all hits).
        feather_path : pathlib.Path or None, optional
            If not None, path to write the DataFrame in Feather format.
        parquet_path : pathlib.Path or None, optional
            If not None, path to write the DataFrame in Parquet format.
        root_path : pathlib.Path or None, optional
            If not None, path to write the DataFrame to a ROOT file.
        root_tree : str, optional
            Name of the TTree inside the ROOT file (default: `"sampic_hits"`).
        extra_header_bytes : int, optional
            Number of bytes to include *after* the detected header boundary
            (default: 1 to capture the trailing newline).
        chunk_size : int, optional
            Byte size for each memory-mapped file read chunk
            (default: 64 KiB).
        batch_size : int, optional
            Number of records to collect before flushing to output
            (default: 100 000).

        Raises
        ------
        ValueError
            If header parsing fails, or if writing to any format encounters
            missing or mismatched branch/column schemas.

        Notes
        -----
        - Feather and Parquet outputs preserve exact column dtypes by
          casting before writing.
        - ROOT output is written via `uproot` using dict-of-NumPy-arrays
          (or `mktree` + `extend`) to ensure correct branch types.
        - Each batch is written immediately; the final partial batch is
          flushed at the end.
        """
        buffer: list[dict] = []
        first = True

        # Schema related objects
        schema = None

        # Writers placeholders
        parquet_writer = None
        feather_writer = None
        root_tree_obj = None

        for hit_record in self.parse_hit_records(
            limit_hits=limit_hits, extra_header_bytes=extra_header_bytes, chunk_size=chunk_size, debug=debug
        ):
            buffer.append(hit_record)
            if len(buffer) < batch_size:
                continue

            df_batch = pd.DataFrame(buffer)
            convert_df_with_schema(df_batch)
            if first:
                schema = build_schema(df_batch.columns)
                schema = schema.with_metadata(self.prepare_header_metadata())
            table = pa.Table.from_pandas(df_batch, schema=schema, preserve_index=False)
            df_batch.reset_index(drop=True, inplace=True)

            # Initialize & write first batch
            root_written = False
            if first:
                if parquet_path:
                    parquet_writer = pq.ParquetWriter(parquet_path, table.schema)
                if feather_path:
                    sink = open(feather_path, "wb")
                    feather_writer = new_file(sink, table.schema)
                if root_path:
                    froot = uproot.recreate(root_path)
                    froot[root_tree] = get_root_data_with_schema(df_batch)  # df_batch.to_dict(orient="list")
                    root_tree_obj = froot[root_tree]
                    root_written = True
                first = False

            # Append subsequent batches
            if parquet_writer:
                parquet_writer.write_table(table)
            if feather_writer:
                feather_writer.write(table)
            if root_tree_obj and not root_written:
                root_tree_obj.extend(get_root_data_with_schema(df_batch))  # df_batch.to_dict(orient="list"))

            buffer.clear()

        # Flush final partial batch
        if buffer:
            df_batch = pd.DataFrame(buffer)
            convert_df_with_schema(df_batch)
            if first:
                schema = build_schema(df_batch.columns)
                schema = schema.with_metadata(self.prepare_header_metadata())
            table = pa.Table.from_pandas(df_batch, schema=schema, preserve_index=False)
            df_batch.reset_index(drop=True, inplace=True)

            root_written = False
            if first:
                if parquet_path:
                    parquet_writer = pq.ParquetWriter(parquet_path, table.schema)
                if feather_path:
                    sink = open(feather_path, "wb")
                    feather_writer = new_file(sink, table.schema)
                if root_path:
                    froot = uproot.recreate(root_path)
                    froot[root_tree] = get_root_data_with_schema(df_batch)  # df_batch.to_dict(orient="list")
                    root_tree_obj = froot[root_tree]
                    root_written = True
                first = False

            if parquet_writer:
                parquet_writer.write_table(table)
            if feather_writer:
                feather_writer.write(table)
            if root_tree_obj and not root_written:
                root_tree_obj.extend(get_root_data_with_schema(df_batch))  # df_batch.to_dict(orient="list"))

        if root_path:
            self.write_root_header(froot)

        # This is probably the place to put the trigger file processing

        # Close writers
        if parquet_writer:
            parquet_writer.close()
        if feather_writer:
            feather_writer.close()
            sink.close()
        # ROOT file closed by context of recreate()
