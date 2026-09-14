# -*- coding: utf-8 -*-
#############################################################################
# zlib License
#
# (C) 2026 Cristóvão Beirão da Cruz e Silva <cbeiraod@cern.ch>
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

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import TypeVar

# Define a type variable restricted to int or float
TimeType = TypeVar('TimeType', int, float)


@dataclass(order=True)
class TimestampedRecord:
    """
    Container for a hit record with an associated timestamp, sortable by time.

    Attributes
    ----------
    timestamp : TimeType
        The hit timestamp. Use `int` for exact picoseconds,
        or `float` for legacy second (since epoch or reconstructed) configurations.
    fine_timestamp : float
        The sub-ps timestamp, when using the updated hit time logic implementing time
        in ps counter and separate fine grain timing for sub ps timing
    channel : int
        The channel where the hit was recorded
    record : Any
        The full hit data (e.g., dict of field values).  Not used for ordering.
    """

    timestamp: TimeType
    fine_timestamp: float
    channel: int
    record: Any = field(compare=False)


@dataclass(order=True)
class WaveformRecord:
    """
    Container for SAMPIC waveform data, sortable uniquely by hit number.

    Attributes
    ----------
    hit_number : int
        The unique identifier or chronological sequence number of the hit.
        Used as the sole key for sorting and comparisons.
    channel : int
        The physical or logical channel that recorded the waveform.
    samples : int
        The total number of digitized samples in the waveform.
    data_samples : list of float
        The array of voltage or ADC values representing the waveform.
    first_cell_index : int
        The physical index of the first sampling cell in the switched capacitor
        array (e.g., the ring buffer starting position).
    trigger_samples : int or None
        The number of samples associated with the trigger threshold crossing,
        if available.
    baseline : float or None
        The calculated baseline offset for the waveform, if available.
    """

    hit_number: int
    channel: int = field(compare=False)
    samples: int = field(compare=False)
    data_samples: list[float] = field(compare=False)
    first_cell_index: int = field(compare=False)
    trigger_samples: int | None = field(compare=False)
    baseline: float | None = field(compare=False)
