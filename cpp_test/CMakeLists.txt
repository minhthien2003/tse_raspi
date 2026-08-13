cmake_minimum_required(VERSION 3.16)
project(v49_pi5_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)

pkg_check_modules(FFTW3F REQUIRED IMPORTED_TARGET fftw3f)
pkg_check_modules(SNDFILE REQUIRED IMPORTED_TARGET sndfile)
pkg_check_modules(PORTAUDIO REQUIRED IMPORTED_TARGET portaudio-2.0)

# ONNX Runtime
set(ONNXRUNTIME_INCLUDE_DIR
    "/usr/include/onnxruntime"
)

set(ONNXRUNTIME_LIBRARY
    "/usr/lib/aarch64-linux-gnu/libonnxruntime.so.1.21.0"
)

add_executable(v49_pi5
    v49_pi5.cpp
)

target_include_directories(v49_pi5 PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(v49_pi5 PRIVATE
    ${ONNXRUNTIME_LIBRARY}
    PkgConfig::FFTW3F
    PkgConfig::SNDFILE
    PkgConfig::PORTAUDIO
    pthread
    dl
)

target_compile_options(v49_pi5 PRIVATE
    -O3
    -march=armv8-a
)