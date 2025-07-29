# Copyright 2020 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

########################################################################
# Check if there is C++ code at all
########################################################################
if(NOT rxprocessing_ecological_sources)
    message(STATUS "No C++ sources... skipping python bindings")
    return()
endif(NOT rxprocessing_ecological_sources)

########################################################################
# Check for pygccxml
########################################################################
gr_python_check_module_raw("pygccxml" "import pygccxml" PYGCCXML_FOUND)

include(GrPybind)

########################################################################
# Python Bindings
########################################################################

list(APPEND rxprocessing_ecological_python_files
    rrcpulseshaping_python.cc python_bindings.cc)

gr_pybind_make_oot(rxprocessing_ecological ../../.. gr::rxprocessing_ecological "${rxprocessing_ecological_python_files}")

# copy bindings extension for use in QA test module
add_custom_command(
    TARGET rxprocessing_ecological_python
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:rxprocessing_ecological_python>
            ${PROJECT_BINARY_DIR}/test_modules/gnuradio/rxprocessing_ecological/)

install(
    TARGETS rxprocessing_ecological_python
    DESTINATION ${GR_PYTHON_DIR}/gnuradio/rxprocessing_ecological
    COMPONENT pythonapi)
