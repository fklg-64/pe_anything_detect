#!/usr/bin/env bash
# cppsrc のビルドスクリプト（libpe_segmentation.so / libpe_segmentation.a / サンプル / テスト）。
#
# 使い方:
#   ./build.sh                 Release ビルド（build/ に出力）
#   ./build.sh --debug         Debug ビルド
#   ./build.sh --clean         build/ を消してからビルド
#   ./build.sh --test          ビルド後に単体テストを実行
#   ./build.sh --install DIR   ビルド後に DIR へ install（lib/・include/・bin/）
#   ./build.sh --no-examples --no-tests --shared-only --static-only --jobs N
#
# OpenVINO は /opt/intel/openvino を自動で探す。別の場所なら OPENVINO_DIR（runtime/cmake のあるディレクトリ、
# または OpenVINOConfig.cmake のあるディレクトリ）で指定する。OpenCV は pkg-config / cmake が見つけたものを使う。
#   例) OPENVINO_DIR=/opt/intel/openvino_2026/runtime/cmake ./build.sh
# Python 版と突き合わせるテストも走らせるには、PES_TEST_DATA_DIR=<scripts/gen_*.py で作ったデータのディレクトリ> ./build.sh --test
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

build_type=Release
build_dir=build
clean=0
run_tests=0
install_prefix=""
extra_cmake_args=()
jobs="$(nproc 2>/dev/null || echo 4)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug) build_type=Debug ;;
    --release) build_type=Release ;;
    --clean) clean=1 ;;
    --test) run_tests=1 ;;
    --install) install_prefix="${2:?--install にはディレクトリが必要です}"; shift ;;
    --jobs|-j) jobs="${2:?--jobs には数が必要です}"; shift ;;
    --no-examples) extra_cmake_args+=(-DPES_BUILD_EXAMPLES=OFF) ;;
    --no-tests) extra_cmake_args+=(-DPES_BUILD_TESTS=OFF) ;;
    --shared-only) extra_cmake_args+=(-DPES_BUILD_STATIC=OFF) ;;
    --static-only) extra_cmake_args+=(-DPES_BUILD_SHARED=OFF) ;;
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "不明なオプション: $1（--help を参照）" >&2; exit 2 ;;
  esac
  shift
done

# Python 版と突き合わせるテストの正解データ（README の「テスト」参照）。kmeans/ と reference/ を置いたディレクトリ
if [[ -n "${PES_TEST_DATA_DIR:-}" ]]; then
  extra_cmake_args+=(-DPES_TEST_DATA_DIR="${PES_TEST_DATA_DIR}")
fi

# OpenVINO の場所
if [[ -n "${OPENVINO_DIR:-}" ]]; then
  if [[ -f "${OPENVINO_DIR}/OpenVINOConfig.cmake" ]]; then
    extra_cmake_args+=(-DOpenVINO_DIR="${OPENVINO_DIR}")
  else
    extra_cmake_args+=(-DOpenVINO_DIR="${OPENVINO_DIR}/runtime/cmake")
  fi
fi

if [[ "${clean}" == 1 ]]; then
  rm -rf "${build_dir}"
fi

cmake -S . -B "${build_dir}" -DCMAKE_BUILD_TYPE="${build_type}" "${extra_cmake_args[@]}"
cmake --build "${build_dir}" -j "${jobs}"

if [[ "${run_tests}" == 1 ]]; then
  (cd "${build_dir}" && ctest --output-on-failure)
fi

if [[ -n "${install_prefix}" ]]; then
  cmake --install "${build_dir}" --prefix "${install_prefix}"
fi

echo
echo "ビルド完了 (${build_type}):"
ls -1 "${build_dir}"/libpe_segmentation* "${build_dir}"/bin/* 2>/dev/null | sed 's/^/  /' || true
