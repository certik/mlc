cmake_minimum_required(VERSION 3.27)
project(mlc_clib C)

set(CMAKE_C_STANDARD 11)

add_executable(mlc_clib
    main.c
    kernels.c
    gguf.c
    display.c
    inference-generated.c
)

target_link_libraries(mlc_clib m)

add_executable(mlc_clib2
    main2.c
    kernels.c
    gguf.c
    display.c
    inference-generated2.c
)

target_link_libraries(mlc_clib2 m)
