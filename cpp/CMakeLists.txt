cmake_minimum_required(VERSION 3.16)
project(tse_voice_lock CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# ONNX Runtime
#
# Tai ban aarch64 tu github.com/microsoft/onnxruntime/releases roi giai nen,
# sau do tro ONNXRUNTIME_ROOT vao thu muc do:
#
#   cmake -B build -DONNXRUNTIME_ROOT=$HOME/onnxruntime-linux-aarch64-1.20.1
#
# Neu da cai vao he thong (/usr/local) thi khong can dat bien nay.
# ---------------------------------------------------------------------------

set(ONNXRUNTIME_ROOT "" CACHE PATH "Thu muc onnxruntime da giai nen")

find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_ROOT}/include
  PATHS /usr/local/include/onnxruntime /usr/include/onnxruntime
        /usr/local/include /usr/include
  PATH_SUFFIXES core/session
)

find_library(ORT_LIBRARY NAMES onnxruntime
  HINTS ${ONNXRUNTIME_ROOT}/lib
  PATHS /usr/local/lib /usr/lib /usr/lib/aarch64-linux-gnu
)

if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
  message(FATAL_ERROR
    "Khong tim thay ONNX Runtime.\n"
    "  Tai ban aarch64:\n"
    "    wget https://github.com/microsoft/onnxruntime/releases/download/"
    "v1.20.1/onnxruntime-linux-aarch64-1.20.1.tgz\n"
    "    tar xf onnxruntime-linux-aarch64-1.20.1.tgz\n"
    "  Roi build lai voi:\n"
    "    cmake -B build -DONNXRUNTIME_ROOT=$PWD/onnxruntime-linux-aarch64-1.20.1")
endif()

message(STATUS "ONNX Runtime include: ${ORT_INCLUDE_DIR}")
message(STATUS "ONNX Runtime lib    : ${ORT_LIBRARY}")

# ---------------------------------------------------------------------------
# ALSA
# ---------------------------------------------------------------------------

find_package(PkgConfig REQUIRED)
pkg_check_modules(ALSA REQUIRED alsa)

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Muc tieu
# ---------------------------------------------------------------------------

add_executable(tse_voice_lock
  src/main.cpp
  src/core.cpp
  src/io.cpp
  src/voice_lock.cpp
  src/enroll.cpp
  src/audio.cpp
)

target_include_directories(tse_voice_lock PRIVATE
  ${ORT_INCLUDE_DIR}
  ${ALSA_INCLUDE_DIRS}
)

target_link_libraries(tse_voice_lock PRIVATE
  ${ORT_LIBRARY}
  ${ALSA_LIBRARIES}
  Threads::Threads
)

target_compile_options(tse_voice_lock PRIVATE
  -Wall -Wextra -Wno-unused-parameter
  $<$<CONFIG:Release>:-O3>
)

# Cortex-A76 cua Pi 5. Bo dong nay neu build cho CPU khac.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mcpu=cortex-a76" HAS_A76)
  if(HAS_A76)
    target_compile_options(tse_voice_lock PRIVATE -mcpu=cortex-a76)
    message(STATUS "Bat -mcpu=cortex-a76 (Pi 5)")
  endif()
endif()

# Nhung rpath de chay duoc ma khong can dat LD_LIBRARY_PATH
if(ONNXRUNTIME_ROOT)
  set_target_properties(tse_voice_lock PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
    INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib"
  )
endif()

install(TARGETS tse_voice_lock RUNTIME DESTINATION bin)
