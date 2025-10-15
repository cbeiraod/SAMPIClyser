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

from __future__ import annotations

__version__ = '0.1.1'

from .sampic_decoder import SAMPIC_Run_Decoder
from .sampic_tools import check_time_ordering
from .sampic_tools import get_channel_hits
from .sampic_tools import get_file_metadata
from .sampic_tools import plot_channel_hit_rate
from .sampic_tools import plot_channel_hits
from .sampic_tools import plot_channel_waveforms
from .sampic_tools import plot_hit_rate
from .sampic_tools import set_mplhep_style
from .sensor_hitmaps import plot_hitmap

__all__ = [
    "SAMPIC_Run_Decoder",
    "set_mplhep_style",
    "get_channel_hits",
    "plot_channel_hits",
    "plot_hit_rate",
    "plot_channel_hit_rate",
    "plot_hitmap",
    "plot_channel_waveforms",
    "check_time_ordering",
    "get_file_metadata",
]
