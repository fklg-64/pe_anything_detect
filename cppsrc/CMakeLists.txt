cmake_minimum_required(VERSION 3.16)
project(pe_segmentation VERSION 1.0.0 LANGUAGES CXX)

# PE-Spatial CLIP + EdgeTAM のセマンティックセグメンテーション（OpenVINO / C++ 版）。
# libpe_segmentation.so（共有）と libpe_segmentation.a（静的）を作る。使い方は README.md。

option(PES_BUILD_SHARED "共有ライブラリ (libpe_segmentation.so) を作る" ON)
option(PES_BUILD_STATIC "静的ライブラリ (libpe_segmentation.a) を作る" ON)
option(PES_BUILD_EXAMPLES "サンプル (pe_segmentation_sample) を作る" ON)
option(PES_BUILD_TESTS "単体テストを作る" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "ビルドの種類" FORCE)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# OpenVINO の C++ SDK。OpenVINO_DIR を指定するか、setupvars.sh を読み込み済みなら INTEL_OPENVINO_DIR から探す
if(NOT OpenVINO_DIR)
  foreach(candidate "$ENV{INTEL_OPENVINO_DIR}/runtime/cmake" "/opt/intel/openvino/runtime/cmake" "/opt/intel/openvino_2026/runtime/cmake")
    if(EXISTS "${candidate}/OpenVINOConfig.cmake")
      set(OpenVINO_DIR "${candidate}" CACHE PATH "OpenVINO の cmake ディレクトリ")
      break()
    endif()
  endforeach()
endif()
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
message(STATUS "OpenVINO ${OpenVINO_VERSION} (${OpenVINO_DIR}) / OpenCV ${OpenCV_VERSION}")

# --- BPE の merges を埋め込む（実行時にファイルを探さなくてよいように） ---
set(PES_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(PES_MERGES_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/data/clip_bpe_merges.txt")
set(PES_MERGES_CPP "${PES_GENERATED_DIR}/clip_bpe_merges_data.cpp")
add_custom_command(
  OUTPUT "${PES_MERGES_CPP}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${PES_GENERATED_DIR}"
  COMMAND ${CMAKE_COMMAND} -DINPUT=${PES_MERGES_SOURCE} -DOUTPUT=${PES_MERGES_CPP} -DSYMBOL=pes_clip_bpe_merges
          -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_file.cmake"
  DEPENDS "${PES_MERGES_SOURCE}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_file.cmake"
  COMMENT "BPE の merges を埋め込みデータにする"
  VERBATIM)

set(PES_SOURCES
  src/api.cpp
  src/clip_extractor.cpp
  src/coarse_segmentation.cpp
  src/config.cpp
  src/edgetam_runner.cpp
  src/kmeans.cpp
  src/mask_utils.cpp
  src/ov_common.cpp
  src/segmenter.cpp
  src/semantic_refiner.cpp
  src/simd.cpp
  src/simd_avx2.cpp
  src/tokenizer.cpp
  src/yaml_lite.cpp
  "${PES_MERGES_CPP}")

# --- コンパイルオプション ---
# 浮動小数点は NumPy と同じ演算順序にそろえる（FMA への自動置き換えをしない。FMA は simd_avx2.cpp の中で明示的に使う）
set(PES_COMMON_FLAGS -O3 -ffp-contract=off -fvisibility=hidden -fvisibility-inlines-hidden -Wall -Wextra)
# AVX2 の実装だけ -mavx2 -mfma（他は汎用の命令。実行時に CPU を調べて切り替える）
set_source_files_properties(src/simd_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")

# 全ターゲットで共通のオブジェクト（共有・静的の両方に同じものを入れる）
add_library(pes_objects OBJECT ${PES_SOURCES})
target_include_directories(pes_objects PUBLIC include PRIVATE src)
target_include_directories(pes_objects SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
target_compile_options(pes_objects PRIVATE ${PES_COMMON_FLAGS})
target_link_libraries(pes_objects PUBLIC openvino::runtime ${OpenCV_LIBS})
set_target_properties(pes_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(pes_objects PRIVATE PES_BUILDING_SHARED)

set(PES_INSTALL_TARGETS "")
if(PES_BUILD_SHARED)
  add_library(pe_segmentation SHARED $<TARGET_OBJECTS:pes_objects>)
  target_link_libraries(pe_segmentation PRIVATE pes_objects)
  target_include_directories(pe_segmentation PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
  set_target_properties(pe_segmentation PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 1 CXX_VISIBILITY_PRESET hidden)
  # 外に出すのは pes_* の C API だけ
  target_link_options(pe_segmentation PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cmake/pes_exports.map")
  set_target_properties(pe_segmentation PROPERTIES LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/pes_exports.map")
  list(APPEND PES_INSTALL_TARGETS pe_segmentation)
endif()
if(PES_BUILD_STATIC)
  add_library(pe_segmentation_static STATIC $<TARGET_OBJECTS:pes_objects>)
  target_link_libraries(pe_segmentation_static PUBLIC openvino::runtime ${OpenCV_LIBS})
  target_include_directories(pe_segmentation_static PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
  set_target_properties(pe_segmentation_static PROPERTIES OUTPUT_NAME pe_segmentation)
  list(APPEND PES_INSTALL_TARGETS pe_segmentation_static)
endif()
if(NOT PES_BUILD_SHARED AND NOT PES_BUILD_STATIC)
  message(FATAL_ERROR "PES_BUILD_SHARED か PES_BUILD_STATIC のどちらかは ON にしてください")
endif()

# 以降のサンプル・テストが使うライブラリ（共有があれば共有）
if(PES_BUILD_SHARED)
  set(PES_LINK_TARGET pe_segmentation)
else()
  set(PES_LINK_TARGET pe_segmentation_static)
endif()

# --- サンプル ---
if(PES_BUILD_EXAMPLES)
  add_executable(pe_segmentation_sample examples/main.cpp)
  target_link_libraries(pe_segmentation_sample PRIVATE ${PES_LINK_TARGET})
  target_include_directories(pe_segmentation_sample SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(pe_segmentation_sample PRIVATE ${OpenCV_LIBS})
  target_compile_options(pe_segmentation_sample PRIVATE -O2 -Wall -Wextra)
  list(APPEND PES_INSTALL_TARGETS pe_segmentation_sample)
endif()

# --- テスト ---
if(PES_BUILD_TESTS)
  enable_testing()
  # 内部の関数を直接試すので、オブジェクトを直接リンクする
  function(pes_add_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE pes_objects)
    target_include_directories(${name} PRIVATE src tests)
    target_include_directories(${name} SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_compile_definitions(${name} PRIVATE PES_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_options(${name} PRIVATE -O2 -ffp-contract=off -Wall -Wextra)
    set_target_properties(${name} PROPERTIES CXX_VISIBILITY_PRESET default)
  endfunction()
  pes_add_test(test_simd tests/test_simd.cpp)
  pes_add_test(test_tokenizer tests/test_tokenizer.cpp)
  pes_add_test(test_config tests/test_config.cpp)
  pes_add_test(test_kmeans tests/test_kmeans.cpp)
  pes_add_test(test_reference tests/test_reference.cpp)
  # C API の通しのテスト（公開ヘッダーと共有ライブラリだけを使う。モデルが必要）
  add_executable(test_api tests/test_api.cpp)
  target_link_libraries(test_api PRIVATE ${PES_LINK_TARGET} ${OpenCV_LIBS})
  target_include_directories(test_api SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
  add_test(NAME simd COMMAND test_simd)
  add_test(NAME tokenizer COMMAND test_tokenizer)
  add_test(NAME config COMMAND test_config)
  add_test(NAME api COMMAND test_api ${CMAKE_CURRENT_SOURCE_DIR}/..)
  # 次の 2 つは Python（faiss / リポジトリの core）で作った正解データが必要。scripts/gen_*.py で作り、PES_TEST_DATA_DIR に置く
  set(PES_TEST_DATA_DIR "" CACHE PATH "scripts/gen_kmeans_cases.py / gen_reference_cases.py が作ったデータのディレクトリ")
  if(PES_TEST_DATA_DIR)
    add_test(NAME kmeans COMMAND test_kmeans ${PES_TEST_DATA_DIR}/kmeans)
    add_test(NAME reference COMMAND test_reference ${PES_TEST_DATA_DIR}/reference)
  endif()
endif()

# --- インストール ---
include(GNUInstallDirs)
install(TARGETS ${PES_INSTALL_TARGETS}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES include/pe_segmentation.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
