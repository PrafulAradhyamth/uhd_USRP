#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 praful.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#


import numpy as np
from gnuradio import gr

class matchedFilter(gr.basic_block):
    """
    A simple matched filter block:
    - Convolves incoming float32 signal with a filter tap (impulse response).
    - Outputs the result with optional thresholding.
    """

    def __init__(self, taps, threshold=0.0, verbose=False):
        gr.basic_block.__init__(self,
            name="matchedFilter",
            in_sig=[np.float32],
            out_sig=[np.float32])

        # Store parameters
        self.taps = np.array(taps, dtype=np.float32)[::-1]  # Flip taps for matched filter
        self.threshold = threshold
        self.verbose = verbose

        # Buffer to store leftover samples between calls
        self.input_buffer = np.array([], dtype=np.float32)

    def general_work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]

        # Append new input to leftover from last call
        self.input_buffer = np.concatenate((self.input_buffer, in0))

        # Do convolution only if we have enough samples
        if len(self.input_buffer) < len(self.taps):
            self.consume(0, len(in0))  # consume all, but don't produce anything
            return 0

        # Full convolution result
        result = np.convolve(self.input_buffer, self.taps, mode='valid')

        # Trim result if it's longer than output buffer
        noutput_items = min(len(out), len(result))
        out[:noutput_items] = result[:noutput_items]

        # Keep leftover samples for next call
        self.input_buffer = self.input_buffer[noutput_items:]

        # Consume the input we used
        self.consume(0, len(in0))
        return noutput_items

