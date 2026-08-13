cmake_minimum_required(VERSION 3.16)
project(v49_pi5_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(PkgConfig REQUIRED)

pkg_check_modules(FFTW3F REQUIRED IMPORTED_TARGET fftw3f)
pkg_check_modules(SNDFILE REQUIRED IMPORTED_TARGET sndfile)
pkg_check_modules(PORTAUDIO REQUIRED IMPORTED_TARGET portaudio-2.0)

find_package(Threads REQUIRED)

# ============================================================
# ONNX Runtime 1.29.0 (aarch64)
#
# Mac dinh dung ban da giai nen trong repo:
#   <repo>/onnxruntime-linux-aarch64-1.29.0
# Ghi de bang:
#   cmake -B build -DONNXRUNTIME_ROOT=/duong/dan/khac
# ============================================================

set(ONNXRUNTIME_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime-linux-aarch64-1.29.0"
    CACHE PATH "Thu muc ONNX Runtime da giai nen"
)

find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_ROOT}/include
  PATHS /usr/local/include/onnxruntime /usr/include/onnxruntime
        /usr/local/include /usr/include
  PATH_SUFFIXES core/session
  NO_DEFAULT_PATH
)
find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h)

find_library(ORT_LIBRARY NAMES onnxruntime
  HINTS ${ONNXRUNTIME_ROOT}/lib
  PATHS /usr/local/lib /usr/lib /usr/lib/aarch64-linux-gnu
  NO_DEFAULT_PATH
)
find_library(ORT_LIBRARY NAMES onnxruntime)

if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
  message(FATAL_ERROR
    "Khong tim thay ONNX Runtime.\n"
    "  Tai ban aarch64 1.29.0:\n"
    "    wget https://github.com/microsoft/onnxruntime/releases/download/v1.29.0/onnxruntime-linux-aarch64-1.29.0.tgz\n"
    "    tar xf onnxruntime-linux-aarch64-1.29.0.tgz\n"
    "  Roi build lai voi:\n"
    "    cmake -B build -DONNXRUNTIME_ROOT=$PWD/onnxruntime-linux-aarch64-1.29.0")
endif()

message(STATUS "ONNX Runtime root   : ${ONNXRUNTIME_ROOT}")
message(STATUS "ONNX Runtime include: ${ORT_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ORT_LIBRARY}")

add_executable(v49_pi5
    v49_pi5.cpp
)

target_include_directories(v49_pi5 PRIVATE
    ${ORT_INCLUDE_DIR}
)

target_link_libraries(v49_pi5 PRIVATE
    ${ORT_LIBRARY}
    PkgConfig::FFTW3F
    PkgConfig::SNDFILE
    PkgConfig::PORTAUDIO
    Threads::Threads
    dl
)

target_compile_options(v49_pi5 PRIVATE
    $<$<CONFIG:Release>:-O3>
)

# Cortex-A76 cua Pi 5. Tu dong bo qua neu compiler khong ho tro.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mcpu=cortex-a76" HAS_A76)
  if(HAS_A76)
    target_compile_options(v49_pi5 PRIVATE -mcpu=cortex-a76)
    message(STATUS "Bat -mcpu=cortex-a76 (Pi 5)")
  else()
    target_compile_options(v49_pi5 PRIVATE -march=armv8-a)
  endif()
endif()

# Nhung rpath de chay duoc ma khong can dat LD_LIBRARY_PATH
get_filename_component(ORT_LIB_DIR "${ORT_LIBRARY}" DIRECTORY)
set_target_properties(v49_pi5 PROPERTIES
  BUILD_RPATH "${ORT_LIB_DIR}"
  INSTALL_RPATH "${ORT_LIB_DIR}"
)

install(TARGETS v49_pi5 RUNTIME DESTINATION bin)
