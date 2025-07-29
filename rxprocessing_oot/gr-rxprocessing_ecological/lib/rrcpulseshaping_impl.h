/* -*- c++ -*- */
/*
 * Copyright 2025 praful.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_RXPROCESSING_ECOLOGICAL_RRCPULSESHAPING_IMPL_H
#define INCLUDED_RXPROCESSING_ECOLOGICAL_RRCPULSESHAPING_IMPL_H

#include <gnuradio/rxprocessing_ecological/rrcpulseshaping.h>

namespace gr {
namespace rxprocessing_ecological {

class rrcpulseshaping_impl : public rrcpulseshaping
{
private:
    // Nothing to declare in this block.

public:
    rrcpulseshaping_impl(int sps, float alpha, int ntaps, float gain);
    ~rrcpulseshaping_impl();

    // Where all the action really happens
    void forecast(int noutput_items, gr_vector_int& ninput_items_required);

    int general_work(int noutput_items,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items);
};

} // namespace rxprocessing_ecological
} // namespace gr

#endif /* INCLUDED_RXPROCESSING_ECOLOGICAL_RRCPULSESHAPING_IMPL_H */
