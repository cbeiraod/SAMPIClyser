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
import struct
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from struct import Struct
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import Tuple

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
    # Name: (pandas, pyarrow, numpy for root)
    "HITNumber": ("int32", pa.int32(), np.int32),
    "UnixTime": ("float64", pa.float64(), np.double),
    "Channel": ("int32", pa.int32(), np.int32),
    "Cell": ("int32", pa.int32(), np.int32),
    "TimeStampA": ("int32", pa.int32(), np.int32),
    "TimeStampB": ("int32", pa.int32(), np.int32),
    "FPGATimeStamp": ("uint64", pa.uint64(), np.double),
    "StartOfADCRamp": ("int32", pa.int32(), np.int32),
    "RawTOTValue": ("int32", pa.int32(), np.int32),
    "TOTValue": ("int32", pa.int32(), np.int32),
    "PhysicalCell0Time": ("float64", pa.float64(), np.double),
    "OrderedCell0Time": ("float64", pa.float64(), np.double),
    "Time": ("float64", pa.float64(), np.double),
    "Baseline": ("float32", pa.float32(), np.float32),
    "RawPeak": ("float32", pa.float32(), np.float32),
    "Amplitude": ("float32", pa.float32(), np.float32),
    "ADCCounterLatched": ("int32", pa.int32(), np.int32),
    "DataSize": ("int32", pa.int32(), np.int32),
    "TriggerPosition": (None, pa.list_(pa.int32()), np.int32),
    "DataSample": (None, pa.list_(pa.float32()), np.float32),
    # … etc …
}


def build_schema(metadata: Dict[bytes, bytes] = None, schemaInfo: Dict[str, Tuple] = SAMPIC_Schema_Info):
    fields = [pa.field(name, schemaInfo[name][1]) for name in schemaInfo if schemaInfo[name][1] is not None]
    schema = pa.schema(fields)

    if metadata is not None:
        schema = schema.with_metadata(metadata)

    return schema


def convert_df_with_schema(df: pd.DataFrame, schemaInfo: Dict[str, Tuple] = SAMPIC_Schema_Info):
    for column in df:
        if column in schemaInfo and schemaInfo[column][0] is not None:
            df[column] = df[column].astype(schemaInfo[column][0])


def get_root_data_with_schema(df: pd.DataFrame, schemaInfo: Dict[str, Tuple] = SAMPIC_Schema_Info):
    try:
        ret_val = {}
        for column in schemaInfo:
            if schemaInfo[column][2] is not None:
                ret_val[column] = np.array(df[column], dtype=schemaInfo[column][2])

        return ret_val
    except ValueError as e:
        print(df["HITNumber"])
        print(df["TriggerPosition"])
        raise e


def build_empty_root_data_with_schema(schemaInfo: Dict[str, Tuple] = SAMPIC_Schema_Info):
    ret_val = {}
    for column in schemaInfo:
        if schemaInfo[column][2] is not None:
            ret_val[column] = np.empty(0, dtype=schemaInfo[column][2])

    return ret_val


@dataclass
class SampicHeader:
    """
    Parsed header metadata from a SAMPIC file.

    Attributes
    ----------
    software_version : str
        Version of the SAMPIC DAQ software.
    timestamp : datetime.datetime
        Run start timestamp as a Python datetime.
    sampic_mezzanine_board_version : str
        Version identifier of the mezzanine board.
    num_channels : int
        Total number of channels in this run.
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
    hit_number_format : str
        Format string for hit numbering.
    unix_time_format : str
        Format string for Unix timestamps.
    data_format : str
        Format string for data values.
    trigger_position_format : str
        Format string for trigger-position values.
    data_samples_format : str
        Format string for the data-sample values.
    inl_correction : bool
        Whether INL correction was applied.
    adc_correction : bool
        Whether ADC correction was applied.
    extra : dict of str → str
        Any unrecognized header fields (key/value both decoded as ASCII).
    """

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
    hit_number_format: str = ""
    unix_time_format: str = ""
    data_format: str = ""
    trigger_position_format: str = ""
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
        self.run_files = natsorted(list(self.run_base_path.glob("*.bin*")))

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
        elif "  " in field:
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
                if val == "NO":
                    val = False
                else:
                    val = True
            elif "INL Correction" == key[-14:]:
                SAMPIC_Run_Decoder._parse_header_field(key[:-19].strip(), header, keep_unparsed=keep_unparsed)
                key = "INL Correction"
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
        elif field[-1] == ']':
            index = field.find('[')

            key = field[:index]
            val = field[index + 1 : -1]

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

        # TODO: Adjust the field_specs according to the header

        # Pre-Compile Struct Formats
        # See https://docs.python.org/3/library/struct.html
        s_i32 = Struct('<i')
        # s_i64 = Struct('<q')
        s_ui64 = Struct('<Q')
        s_f32 = Struct('<f')
        s_f64 = Struct('<d')

        field_specs = [
            # ("HIT number", 4, lambda b: int.from_bytes(b, "little", signed=True), "S"),
            # ("Unix time", 8, lambda b: struct.unpack('d', b)[0], "S"),
            # ("Channel", 4, lambda b: int.from_bytes(b, "little", signed=True), "S"),
            # ("Cell", 4, lambda b: int.from_bytes(b, "little", signed=True), "S"),
            # ("TimeStampA", 4, lambda b: int.from_bytes(b, "little", signed=True), "S"),
            # ("TimeStampB", 4, lambda b: int.from_bytes(b, "little", signed=True), "S"),
            # ("FPGATimeStamp", 8, lambda b: int.from_bytes(b, "little", signed=False), "S"),
            # ("StartOfADCRamp", 4, lambda b: int.from_bytes(b, "little", signed=True), "S"),
            # ("RawTOTValue", 4, lambda b: int.from_bytes(b, "little", signed=True), "S"),
            # ("TOTValue", 4, lambda b: struct.unpack('f', b)[0], "S"),
            # ("PhysicalCell0Time", 8, lambda b: struct.unpack('d', b)[0], "S"),
            # ("OrderedCell0Time", 8, lambda b: struct.unpack('d', b)[0], "S"),
            # ("Time", 8, lambda b: struct.unpack('d', b)[0], "S"),
            # ("Baseline", 4, lambda b: struct.unpack('f', b)[0], "S"),
            # ("RawPeak", 4, lambda b: struct.unpack('f', b)[0], "S"),
            # ("Amplitude", 4, lambda b: struct.unpack('f', b)[0], "S"),
            # ("ADCCounterLatched", 4, lambda b: int.from_bytes(b, "little", signed=True), "S"),
            # ("DataSize", 4, lambda b: int.from_bytes(b, "little", signed=True), "S"),
            # ("TriggerPosition", 4, lambda b: int.from_bytes(b, "little", signed=True), "A[DataSize]"),
            # ("DataSample", 4, lambda b: struct.unpack('f', b)[0], "A[DataSize]"),
            # ("field1", 4, lambda b: int.from_bytes(b, "little", signed=True), "S"),
            # ("field2", 8, lambda b: struct.unpack("<d", b)[0], "S"),
            # ("flag",   1, lambda b: bool(b[0]), "S"),
            # …etc…
            ("HITNumber", 4, lambda v, o: s_i32.unpack_from(v, o)[0], "S"),
            # ("UnixTime", 8, lambda v, o: datetime.fromtimestamp(s_f64.unpack_from(v, o)[0]), "S"),
            ("UnixTime", 8, lambda v, o: s_f64.unpack_from(v, o)[0], "S"),
            ("Channel", 4, lambda v, o: s_i32.unpack_from(v, o)[0], "S"),
            ("Cell", 4, lambda v, o: s_i32.unpack_from(v, o)[0], "S"),
            ("TimeStampA", 4, lambda v, o: s_i32.unpack_from(v, o)[0], "S"),
            ("TimeStampB", 4, lambda v, o: s_i32.unpack_from(v, o)[0], "S"),
            ("FPGATimeStamp", 8, lambda v, o: s_ui64.unpack_from(v, o)[0], "S"),
            ("StartOfADCRamp", 4, lambda v, o: s_i32.unpack_from(v, o)[0], "S"),
            ("RawTOTValue", 4, lambda v, o: s_i32.unpack_from(v, o)[0], "S"),
            ("TOTValue", 4, lambda v, o: s_f32.unpack_from(v, o)[0], "S"),
            ("PhysicalCell0Time", 8, lambda v, o: s_f64.unpack_from(v, o)[0], "S"),
            ("OrderedCell0Time", 8, lambda v, o: s_f64.unpack_from(v, o)[0], "S"),
            ("Time", 8, lambda v, o: s_f64.unpack_from(v, o)[0], "S"),
            ("Baseline", 4, lambda v, o: s_f32.unpack_from(v, o)[0], "S"),
            ("RawPeak", 4, lambda v, o: s_f32.unpack_from(v, o)[0], "S"),
            ("Amplitude", 4, lambda v, o: s_f32.unpack_from(v, o)[0], "S"),
            ("ADCCounterLatched", 4, lambda v, o: s_i32.unpack_from(v, o)[0], "S"),
            ("DataSize", 4, lambda v, o: s_i32.unpack_from(v, o)[0], "S"),
            ("TriggerPosition", 4, lambda v, o: s_i32.unpack_from(v, o)[0], "A[DataSize]"),
            ("DataSample", 4, lambda v, o: s_f32.unpack_from(v, o)[0], "A[DataSize]"),
        ]

        # Helper to try parsing one record from the buffer
        def try_parse_record() -> Dict[str, Any] | None:
            view = memoryview(buffer)

            # First ensure we have at least the fixed portion
            fixed_len = sum(n for _, n, _, t in field_specs if t == "S")
            if len(buffer) < fixed_len:
                return None  # need more data

            # Parse fixed fields to extract the fixed portion, counts should be in this portion so we can parse the rest of the data structure
            record: Dict[str, Any] = {}
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

                # Stream through chunks
                for chunk in body_gen:
                    buffer.extend(chunk)

                    # parse as many complete records as we can
                    while True:
                        if limit_hits > 0 and hits >= limit_hits:
                            return_now = True
                            break
                        rec = try_parse_record()
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
            b'hit_number_format': self.run_header.hit_number_format.encode('ascii'),
            b'unix_time_format': self.run_header.unix_time_format.encode('ascii'),
            b'data_format': self.run_header.data_format.encode('ascii'),
            b'trigger_position_format': self.run_header.trigger_position_format.encode('ascii'),
            b'data_samples_format': self.run_header.data_samples_format.encode('ascii'),
            b'inl_correction': b'\x01' if self.run_header.inl_correction else b'\x00',
            b'adc_correction': b'\x01' if self.run_header.adc_correction else b'\x00',
        }

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
            'hit_number_format': self.run_header.hit_number_format,
            'unix_time_format': self.run_header.unix_time_format,
            'data_format': self.run_header.data_format,
            'trigger_position_format': self.run_header.trigger_position_format,
            'data_samples_format': self.run_header.data_samples_format,
            'inl_correction': self.run_header.inl_correction,
            'adc_correction': self.run_header.adc_correction,
        }

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
        # TODO: Change this to use info from the header
        schema = build_schema()

        # Writers placeholders
        parquet_writer = None
        feather_writer = None
        root_tree_obj = None

        for hit_record in self.parse_hit_records(limit_hits=limit_hits, extra_header_bytes=extra_header_bytes, chunk_size=chunk_size):
            buffer.append(hit_record)
            if len(buffer) < batch_size:
                continue

            df_batch = pd.DataFrame(buffer)
            convert_df_with_schema(df_batch)
            if first:
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

        # Close writers
        if parquet_writer:
            parquet_writer.close()
        if feather_writer:
            feather_writer.close()
            sink.close()
        # ROOT file closed by context of recreate()
