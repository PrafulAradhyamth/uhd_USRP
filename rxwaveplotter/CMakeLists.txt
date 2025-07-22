cmake_minimum_required(VERSION 3.14)
# cmake -DCMAKE_PREFIX_PATH=/path/to/Qt6 ..
project(rxwaveplotter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 COMPONENTS Charts REQUIRED)

add_executable(rxwaveplotter main.cpp)

target_link_libraries(rxwaveplotter PRIVATE Qt6::Charts)