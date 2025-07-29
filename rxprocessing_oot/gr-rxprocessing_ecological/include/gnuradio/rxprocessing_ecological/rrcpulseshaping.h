/* -*- c++ -*- */
/*
 * Copyright 2025 praful.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_RXPROCESSING_ECOLOGICAL_RRCPULSESHAPING_H
#define INCLUDED_RXPROCESSING_ECOLOGICAL_RRCPULSESHAPING_H

#include <gnuradio/block.h>
#include <gnuradio/rxprocessing_ecological/api.h>

namespace gr {
namespace rxprocessing_ecological {

/*!
 * \brief <+description of block+>
 * \ingroup rxprocessing_ecological
 *
 */
class RXPROCESSING_ECOLOGICAL_API rrcpulseshaping : virtual public gr::block
{
public:
    typedef std::shared_ptr<rrcpulseshaping> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of
     * rxprocessing_ecological::rrcpulseshaping.
     *
     * To avoid accidental use of raw pointers, rxprocessing_ecological::rrcpulseshaping's
     * constructor is in a private implementation
     * class. rxprocessing_ecological::rrcpulseshaping::make is the public interface for
     * creating new instances.
     */
    static sptr make(int sps, float alpha, int ntaps, float gain = 1.0);
};

} // namespace rxprocessing_ecological
} // namespace gr

#endif /* INCLUDED_RXPROCESSING_ECOLOGICAL_RRCPULSESHAPING_H */
