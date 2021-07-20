# -*- coding: utf-8 -*-
#
# pyPnF
# A Package for Point and Figure Charting
# https://github.com/swaschke/pypnf
#
# Copyright (C) 2021  Stefan Waschke
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
The testdata module contains different sets of testdata.
"""

def dataset(set):
    """
    testdata
    """

    if set == 'Set 1':

        close = [110, 111, 112, 111, 113, 115, 114, 112, 113, 114, 116, 114, 113, 115, 116, 117]

        ts = {'close': close}

    return ts
