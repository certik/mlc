cmake_minimum_required(VERSION 3.27)
project(mlc_clib C)

set(CMAKE_C_STANDARD 11)

add_executable(mlc_clib
        main.c
        kernels.h
        kernels.c
        gguf.h
        gguf.c)
