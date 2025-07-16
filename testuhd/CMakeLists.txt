cmake_minimum_required(VERSION 3.10)
project(testuhd LANGUAGES CXX)

find_package(UHD REQUIRED)

add_executable(testuhd_ex1 testuhd_ex1.cpp)

target_link_libraries(testuhd_ex1 uhd)

set_target_properties(testuhd_ex1 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/out"
)
