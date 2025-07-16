#
cmake_minimum_required(VERSION 3.10)
project(capdata_uhd LANGUAGES CXX)
find_package(UHD REQUIRED)
add_executable(capture_save_data capture_save_data.cpp)
target_link_libraries(capture_save_data uhd)
set_target_properties(capture_save_data PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/out"
)
