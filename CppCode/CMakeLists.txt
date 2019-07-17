cmake_minimum_required(VERSION 3.12)
project(udnn_team5)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(udnn_team5
        main.cpp data_loader.cpp data_loader.h train.h)
