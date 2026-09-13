cmake_minimum_required(VERSION 3.24)
project(gpt_trainer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# 依赖目录（全部使用绝对路径 / 本地相对路径，不依赖系统 PATH）
#   libtorch   : 本目录解压出的 libtorch-win-shared-with-deps-2.14.0+cu130
#   Qt         : aqtinstall 下载到 qt/6.9.3/msvc2022_64
#   sentencepiece: 复用 cpp分词器 已构建的静态库
#   zlib       : 复用 RTX 项目 vcpkg 构建的 z.lib
# 注意：链接输入使用 E:\cpptrn\* 的 ASCII junction（见 build.ps1 注释），
#   因为 MSVC link.exe 以 ANSI 编码读取 Ninja 的响应文件，中文路径会被破坏。
# ---------------------------------------------------------------------------
list(PREPEND CMAKE_PREFIX_PATH "E:/cpptrn/libtorch")
find_package(Torch REQUIRED)

list(PREPEND CMAKE_PREFIX_PATH "E:/cpptrn/qt/6.9.3/msvc2022_64")
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)

set(SP_ROOT   "E:/project/cpp分词器/sentencepiece-0.2.2")
set(SP_BUILD  "E:/cpptrn/sp")
set(ZLIB_ROOT "E:/cpptrn/zlib")

# 生成项目根路径头文件（UTF-8，避免命令行转义问题）
file(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" _root_native)
file(WRITE "${CMAKE_BINARY_DIR}/gen/root_path.h"
     "#pragma once\n#define GPT_PROJECT_ROOT \"${_root_native}\"\n")

add_executable(gpt_trainer WIN32
  src/main.cpp
  src/config.h     src/config.cpp
  src/json_util.h  src/json_util.cpp
  src/zip_util.h   src/zip_util.cpp
  src/path_util.h  src/path_util.cpp
  src/dataset.h    src/dataset.cpp
  src/model.h      src/model.cpp
  src/legacy_serial.h src/legacy_serial.cpp
  src/tokenizer_worker.h src/tokenizer_worker.cpp
  src/train.h      src/train.cpp
  src/tfevents.h   src/tfevents.cpp
  src/gui.h        src/gui.cpp
)

target_compile_definitions(gpt_trainer PRIVATE NOMINMAX _USE_MATH_DEFINES)
target_include_directories(gpt_trainer PRIVATE
  "${CMAKE_BINARY_DIR}/gen"
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${SP_BUILD}/root/include"
  "${SP_ROOT}/third_party/abseil-cpp"
  "${ZLIB_ROOT}"
)
target_compile_options(gpt_trainer PRIVATE /utf-8 /W3 /EHsc /bigobj /Zc:__cplusplus)

# sentencepiece 静态库 + 其依赖的 abseil 静态库
file(GLOB ABSL_LIBS "${SP_BUILD}/third_party/abseil-cpp/absl/*/Release/*.lib")
file(GLOB ABSL_LIBS2 "${SP_BUILD}/third_party/abseil-cpp/absl/*/Release/*/*.lib")

target_link_libraries(gpt_trainer PRIVATE
  ${TORCH_LIBRARIES}
  Qt6::Core
  Qt6::Gui
  Qt6::Widgets
  "${SP_BUILD}/root/lib/sentencepiece.lib"
  "${SP_BUILD}/root/lib/sentencepiece_train.lib"
  "${ZLIB_ROOT}/z.lib"
  ${ABSL_LIBS}
  ${ABSL_LIBS2}
)

# 复制运行时 DLL（Qt / zlib）到可执行目录
set(QT_BIN "${CMAKE_CURRENT_SOURCE_DIR}/qt/6.9.3/msvc2022_64/bin")
set(QT_PLUGINS "${CMAKE_CURRENT_SOURCE_DIR}/qt/6.9.3/msvc2022_64/plugins")
add_custom_command(TARGET gpt_trainer POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${QT_BIN}/Qt6Core.dll" "${QT_BIN}/Qt6Gui.dll" "${QT_BIN}/Qt6Widgets.dll"
          "$<TARGET_FILE_DIR:gpt_trainer>"
  COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:gpt_trainer>/platforms"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${QT_PLUGINS}/platforms/qwindows.dll" "$<TARGET_FILE_DIR:gpt_trainer>/platforms"
  COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:gpt_trainer>/styles"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${QT_PLUGINS}/styles/qmodernwindowsstyle.dll" "$<TARGET_FILE_DIR:gpt_trainer>/styles"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${ZLIB_ROOT}/z.dll" "$<TARGET_FILE_DIR:gpt_trainer>"
  COMMENT "Deploying Qt/zlib runtime DLLs"
)
