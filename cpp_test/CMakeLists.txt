cmake_minimum_required(VERSION 3.16)
project(v49_pi5_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3F REQUIRED IMPORTED_TARGET fftw3f)
pkg_check_modules(SNDFILE REQUIRED IMPORTED_TARGET sndfile)
pkg_check_modules(PORTAUDIO REQUIRED IMPORTED_TARGET portaudio-2.0)

# Set this to your ONNX Runtime installation prefix.
# Example:
#   cmake -S . -B build -DONNXRUNTIME_ROOT=/opt/onnxruntime
set(ONNXRUNTIME_ROOT "/opt/onnxruntime" CACHE PATH "ONNX Runtime root")

find_path(ONNXRUNTIME_INCLUDE_DIR
    onnxruntime_cxx_api.h
    PATHS
      ${ONNXRUNTIME_ROOT}/include
      ${ONNXRUNTIME_ROOT}/include/onnxruntime
    REQUIRED)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS
      ${ONNXRUNTIME_ROOT}/lib
      ${ONNXRUNTIME_ROOT}/lib64
    REQUIRED)

add_executable(v49_pi5 v49_pi5.cpp)

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

target_compile_options(v49_pi5 PRIVATE -O3 -march=armv8-a)
