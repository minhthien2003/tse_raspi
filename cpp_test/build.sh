#!/usr/bin/env bash
# ============================================================
# Build v49_pi5.cpp tren Raspberry Pi 5 (aarch64, 64-bit OS)
#
#   ./build.sh                 build binh thuong
#   ./build.sh --deps          cai apt package truoc roi build
#   ./build.sh --clean         xoa build/ roi build lai tu dau
#   ./build.sh --ort /path     dung ONNX Runtime o thu muc khac
#   ./build.sh --debug         build Debug thay vi Release
#
# Ket qua: build/v49_pi5
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

ORT_VERSION="1.29.0"
ORT_NAME="onnxruntime-linux-aarch64-${ORT_VERSION}"
ORT_ROOT="${ONNXRUNTIME_ROOT:-${REPO_DIR}/${ORT_NAME}}"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_NAME}.tgz"

BUILD_TYPE="Release"
INSTALL_DEPS=0
DO_CLEAN=0
FETCH_ORT=0
JOBS="$(nproc)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --deps)          INSTALL_DEPS=1; shift ;;
    --clean)         DO_CLEAN=1; shift ;;
    --debug)         BUILD_TYPE="Debug"; shift ;;
    --fetch-ort)     FETCH_ORT=1; shift ;;
    --ort)           ORT_ROOT="$(cd "$2" && pwd)"; shift 2 ;;
    -j)              JOBS="$2"; shift 2 ;;
    -h|--help)       sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Tham so khong hieu: $1" >&2; exit 1 ;;
  esac
done

echo "==> Kiem tra kien truc"
ARCH="$(uname -m)"
if [[ "${ARCH}" != "aarch64" && "${ARCH}" != "arm64" ]]; then
  echo "CANH BAO: kien truc hien tai la '${ARCH}', khong phai aarch64." >&2
  echo "          Thu vien ${ORT_NAME} chi chay tren Pi OS 64-bit." >&2
fi

if [[ ${INSTALL_DEPS} -eq 1 ]]; then
  echo "==> Cai package he thong"
  sudo apt-get update
  sudo apt-get install -y \
    build-essential cmake pkg-config \
    libfftw3-dev libsndfile1-dev portaudio19-dev
fi

echo "==> Kiem tra cong cu build"
MISSING=()
for tool in cmake g++ pkg-config; do
  command -v "${tool}" >/dev/null 2>&1 || MISSING+=("${tool}")
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo "Thieu: ${MISSING[*]}" >&2
  echo "Chay lai voi:  ./build.sh --deps" >&2
  exit 1
fi

echo "==> Kiem tra thu vien dev (pkg-config)"
MISSING_PKG=()
for pkg in fftw3f sndfile portaudio-2.0; do
  pkg-config --exists "${pkg}" || MISSING_PKG+=("${pkg}")
done
if [[ ${#MISSING_PKG[@]} -gt 0 ]]; then
  echo "Thieu thu vien: ${MISSING_PKG[*]}" >&2
  echo "Chay lai voi:  ./build.sh --deps" >&2
  exit 1
fi

echo "==> Kiem tra ONNX Runtime: ${ORT_ROOT}"
if [[ ! -f "${ORT_ROOT}/include/onnxruntime_cxx_api.h" || ! -f "${ORT_ROOT}/lib/libonnxruntime.so" ]]; then
  if [[ ${FETCH_ORT} -eq 1 ]]; then
    echo "    Khong thay, dang tai ${ORT_NAME}..."
    tmp="$(mktemp -d)"
    curl -fL "${ORT_URL}" -o "${tmp}/ort.tgz"
    tar -xzf "${tmp}/ort.tgz" -C "${REPO_DIR}"
    rm -rf "${tmp}"
    ORT_ROOT="${REPO_DIR}/${ORT_NAME}"
  else
    echo "Khong tim thay ONNX Runtime tai: ${ORT_ROOT}" >&2
    echo "Cach 1 - tu dong tai:   ./build.sh --fetch-ort" >&2
    echo "Cach 2 - tai thu cong:" >&2
    echo "    wget ${ORT_URL}" >&2
    echo "    tar xf ${ORT_NAME}.tgz -C ${REPO_DIR}" >&2
    echo "Cach 3 - chi duong dan: ./build.sh --ort /duong/dan/${ORT_NAME}" >&2
    exit 1
  fi
fi

if [[ ${DO_CLEAN} -eq 1 ]]; then
  echo "==> Xoa ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi

echo "==> Cau hinh CMake (${BUILD_TYPE})"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DONNXRUNTIME_ROOT="${ORT_ROOT}"

echo "==> Build (-j${JOBS})"
cmake --build "${BUILD_DIR}" -j"${JOBS}"

echo
echo "Xong: ${BUILD_DIR}/v49_pi5"
echo
echo "Chay thu:"
echo "  ${BUILD_DIR}/v49_pi5 --model ${REPO_DIR}/models/v49_int8.onnx \\"
echo "      --emb speaker_emb.npy --file input.wav -o output.wav"
