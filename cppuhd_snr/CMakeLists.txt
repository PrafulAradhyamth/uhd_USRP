cmake_minimum_required(VERSION 3.10)
project(uhd_snr_example LANGUAGES CXX)

find_package(UHD REQUIRED)

include_directories(include)
add_executable(tx_barker src/tx_barker.cpp )
add_executable(rx_snr src/rx_snr.cpp)
add_executable(txrx_barker_snr src/txrx_barker_snr.cpp)

target_link_libraries(txrx_barker_snr uhd)
target_link_libraries(tx_barker uhd)
target_link_libraries(rx_snr uhd)
