#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 praful.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#


import numpy as np
from gnuradio import gr

class pktdetect(gr.basic_block):
    """
    Basic packet detection block using correlation with a preamble.
    Tags 'pkt_start' when the correlation exceeds a threshold.
    """

    def __init__(self, threshold=0.5, window_len=64, preamble=None, verbose=False):
        gr.basic_block.__init__(self,
            name="pktdetect",
            in_sig=[np.float32],
            out_sig=[np.float32])
        
        self.threshold = threshold
        self.window_len = window_len
        self.verbose = verbose
        if preamble is None:
            # Default preamble: Barker 13
            self.preamble = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1], dtype=np.float32) * 2 - 1
        else:
            self.preamble = np.array(preamble, dtype=np.float32)

        self.preamble_len = len(self.preamble)

    def forecast(self, noutput_items, ninputs):
        return [noutput_items]

    def general_work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]

        ninput = len(in0)
        npreamble = self.preamble_len

        # We only start checking once enough samples have arrived
        noutput = 0
        for i in range(ninput - npreamble):
            window = in0[i:i+npreamble]
            corr = np.dot(window, self.preamble)

            if self.verbose:
                print(f"[{i}] Correlation: {corr:.2f}")

            # Normalize correlation to [0,1] and compare with threshold
            norm_corr = corr / np.dot(self.preamble, self.preamble)
            if norm_corr >= self.threshold:
                # Detected start of a packet!
                self.add_item_tag(0, self.nitems_written(0) + noutput,
                                  gr.pmt.intern("pkt_start"), gr.pmt.PMT_T)
                if self.verbose:
                    print(f"Packet detected at sample {self.nitems_written(0) + noutput}")

                # Output the full window
                out[noutput:noutput+npreamble] = window
                noutput += npreamble
                break  # Stop after first detection for now

        self.consume(0, ninput)
        return noutput


