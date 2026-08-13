cmake_minimum_required(VERSION 3.16)
project(v49_pi5_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)

pkg_check_modules(FFTW3F REQUIRED IMPORTED_TARGET fftw3f)
pkg_check_modules(SNDFILE REQUIRED IMPORTED_TARGET sndfile)
pkg_check_modules(PORTAUDIO REQUIRED IMPORTED_TARGET portaudio-2.0)

# ============================================================
# ONNX Runtime 1.29.0
# ============================================================

set(ONNXRUNTIME_ROOT
    "/home/admin/zena/tse_raspi/onnxruntime-linux-aarch64-1.29.0"
    CACHE PATH "ONNX Runtime root"
)

set(ONNXRUNTIME_INCLUDE_DIR
    "${ONNXRUNTIME_ROOT}/include"
)

set(ONNXRUNTIME_LIBRARY
    "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
)

message(STATUS "ONNX Runtime root   : ${ONNXRUNTIME_ROOT}")
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")

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