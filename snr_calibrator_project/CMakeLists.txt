cmake_minimum_required(VERSION 3.10)
project(snr_calibrator)

find_package(UHD REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW REQUIRED fftw3)

add_executable(snr_calibrator snr_calibrator.cpp)
target_include_directories(snr_calibrator PRIVATE ${FFTW_INCLUDE_DIRS})
target_link_libraries(snr_calibrator uhd fftw3)
