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

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import uproot
from natsort import natsorted
from pyarrow.ipc import new_file
from termcolor import colored


@dataclass
class SampicHeader:
    """Represents the parsed header of a Sampic file"""

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
    front_end_fpga_re = re.compile(r"^FRONT-END FPGA INDEX: (\d+) FIRMWARE VERSION (.+) BASELINE VALUE: ([\d\.]+)")
    timestamp_re = re.compile(r"^UnixTime = (.+) date = (.+) time = (.+ms)")

    def __init__(
        self,
        run_dir_path: Path,
    ):
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
        Context manager that opens `file_path`, mmaps it, extracts the header up to
        the last '=' before the first 0x00 (plus extra_header_bytes), and then
        yields (header_bytes, body_generator). Cleans up file and mmap on exit.

        The function also sets a class member, current_filesize holding the size of the current file

        Args:
            file_path: Path to the binary file.
            extra_header_bytes:   Number of bytes *after* the last header byte to include in the header.
            chunk_size: How many bytes each body‐generator chunk should be.

        Returns:
            (header_bytes, body_generator)
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
        Inspect a single 'field' string and assign to the right dataclass field.
        Unrecognized keys go into header.extra.
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
        Parse the header section (bytes) into a SampicHeader dataclass.

        Header format:
          - Multiple lines.
          - On each line:
            * Starts and ends with "===".
            * Contains one or more fields separated by "===".
            * Field formats vary (e.g. "param: value", "param value", or
              composite "part1 = x part2 = y", etc.)

        Returns:
            A SampicHeader dataclass.
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
    ):
        """
        Decode binary data records from the run.

        Args:
            limit_hits:   stop after yielding this many records.

        Yields:
            A dict (or dataclass) per hit record, mapping field names to values.
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
                            return
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
    ) -> pd.DataFrame:
        """
        Consumes parsed hit-record dicts, builds a pandas DataFrame,
        and writes it out to Feather, Parquet, and/or a ROOT TTree.

        Args:
            limit_hits:   Whether to limit the number of parsed hits to this number (default: 0, no limit)
            feather_path: Path to save as .feather (fast, uncompressed).
            parquet_path: Path to save as .parquet (columnar, compressed).
            root_path:    Path to save as .root file.
            root_tree:    Name of the TTree inside the ROOT file.
            extra_header_bytes: How many extra bytes to add to the header after the end of automatic detection, the default of 1 should always work, since it is adding 1 for the newline character at the end of the header
            chunk_size:   Size of the chunks into which the data file is being split when loaded into memory
            batch_size:   How many hit records to process in one go
        Returns:
            The assembled pandas DataFrame.
        """
        buffer: list[dict] = []
        first = True

        # Schema related objects
        # TODO: Change this to use info from the header
        schema = pa.schema(
            [
                pa.field("HITNumber", pa.int32()),
                pa.field("UnixTime", pa.float64()),
                pa.field("Channel", pa.int32()),
                pa.field("Cell", pa.int32()),
                pa.field("TimeStampA", pa.int32()),
                pa.field("TimeStampB", pa.int32()),
                pa.field("FPGATimeStamp", pa.uint64()),
                pa.field("StartOfADCRamp", pa.int32()),
                pa.field("RawTOTValue", pa.int32()),
                pa.field("TOTValue", pa.int32()),
                pa.field("PhysicalCell0Time", pa.float64()),
                pa.field("OrderedCell0Time", pa.float64()),
                pa.field("Time", pa.float64()),
                pa.field("Baseline", pa.float32()),
                pa.field("RawPeak", pa.float32()),
                pa.field("Amplitude", pa.float32()),
                pa.field("ADCCounterLatched", pa.int32()),
                pa.field("DataSize", pa.int32()),
                pa.field("TriggerPosition", pa.list_(pa.int32())),
                pa.field("DataSample", pa.list_(pa.float32())),
                # … etc …
            ]
        )

        def convert_df_with_schema(df):
            df["HITNumber"] = df["HITNumber"].astype("int32")
            df["UnixTime"] = df["UnixTime"].astype("float64")
            df["Channel"] = df["Channel"].astype("int32")
            df["Cell"] = df["Cell"].astype("int32")
            df["TimeStampA"] = df["TimeStampA"].astype("int32")
            df["TimeStampB"] = df["TimeStampB"].astype("int32")
            df["FPGATimeStamp"] = df["FPGATimeStamp"].astype("uint64")
            df["StartOfADCRamp"] = df["StartOfADCRamp"].astype("int32")
            df["RawTOTValue"] = df["RawTOTValue"].astype("int32")
            df["TOTValue"] = df["TOTValue"].astype("int32")
            df["PhysicalCell0Time"] = df["PhysicalCell0Time"].astype("float64")
            df["OrderedCell0Time"] = df["OrderedCell0Time"].astype("float64")
            df["Time"] = df["Time"].astype("float64")
            df["Baseline"] = df["Baseline"].astype("float32")
            df["RawPeak"] = df["RawPeak"].astype("float32")
            df["Amplitude"] = df["Amplitude"].astype("float32")
            df["ADCCounterLatched"] = df["ADCCounterLatched"].astype("int32")
            df["DataSize"] = df["DataSize"].astype("int32")
            # pa.field("TriggerPosition",  pa.list_(pa.int32())),
            # pa.field("DataSample",  pa.list_(pa.float32())),

        def get_root_data_with_schema(df):
            try:
                ret_val = {
                    "HITNumber": np.array(df["HITNumber"], dtype=np.int32),
                    "UnixTime": np.array(df["UnixTime"], dtype=np.double),
                    "Channel": np.array(df["Channel"], dtype=np.int32),
                    "Cell": np.array(df["Cell"], dtype=np.int32),
                    "TimeStampA": np.array(df["TimeStampA"], dtype=np.int32),
                    "TimeStampB": np.array(df["TimeStampB"], dtype=np.int32),
                    "FPGATimeStamp": np.array(df["FPGATimeStamp"], dtype=np.uint64),
                    "StartOfADCRamp": np.array(df["StartOfADCRamp"], dtype=np.int32),
                    "RawTOTValue": np.array(df["RawTOTValue"], dtype=np.int32),
                    "TOTValue": np.array(df["TOTValue"], dtype=np.int32),
                    "PhysicalCell0Time": np.array(df["PhysicalCell0Time"], dtype=np.double),
                    "OrderedCell0Time": np.array(df["OrderedCell0Time"], dtype=np.double),
                    "Time": np.array(df["Time"], dtype=np.double),
                    "Baseline": np.array(df["Baseline"], dtype=np.float32),
                    "RawPeak": np.array(df["RawPeak"], dtype=np.float32),
                    "Amplitude": np.array(df["Amplitude"], dtype=np.float32),
                    "ADCCounterLatched": np.array(df["ADCCounterLatched"], dtype=np.int32),
                    "DataSize": np.array(df["DataSize"], dtype=np.int32),
                    "TriggerPosition": np.array(df["TriggerPosition"].tolist(), dtype=np.int32),
                    "DataSample": np.array(df["DataSample"].tolist(), dtype=np.float32),
                }

                return ret_val
            except ValueError as e:
                print(df["HITNumber"])
                print(df["TriggerPosition"])
                raise e

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

        # Close writers
        if parquet_writer:
            parquet_writer.close()
        if feather_writer:
            feather_writer.close()
            sink.close()
        # ROOT file closed by context of recreate()
