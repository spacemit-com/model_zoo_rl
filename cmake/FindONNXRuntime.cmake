# FindONNXRuntime.cmake
#
# 查找 ONNX Runtime 头文件和库。支持用户通过变量或环境变量指定路径。
#
# 输入变量（可选）:
#   ONNXRUNTIME_DIR  — ONNX Runtime 安装根目录（CMake 变量或同名环境变量）
#
# 输出变量:
#   ONNXRUNTIME_INCLUDE_DIR  — 头文件目录（含 onnxruntime_cxx_api.h）
#   ONNXRUNTIME_LIB          — libonnxruntime 路径
#   SPACEMIT_EP_LIB          — libspacemit_ep 路径（rv64 apt 安装时自动找到，否则为空）

set(_ort_hints "")
if(DEFINED ONNXRUNTIME_DIR)
    list(APPEND _ort_hints "${ONNXRUNTIME_DIR}")
endif()
if(DEFINED ENV{ONNXRUNTIME_DIR})
    list(APPEND _ort_hints "$ENV{ONNXRUNTIME_DIR}")
endif()
# apt 安装路径（rv64 板卡 /usr，x86_64 开发机 /usr/local）
list(APPEND _ort_hints /usr/local /usr)

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS ${_ort_hints}
    PATH_SUFFIXES
        include/onnxruntime/core/session
        include/onnxruntime
        include
    NO_DEFAULT_PATH
)

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    HINTS ${_ort_hints}
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR
        "ONNX Runtime not found.\n"
        "  apt install (rv64): sudo apt install libonnxruntime-dev\n"
        "  或手动指定: cmake .. -DONNXRUNTIME_DIR=/path/to/onnxruntime")
endif()

message(STATUS "ONNX Runtime: ${ONNXRUNTIME_LIB}")
message(STATUS "  includes:   ${ONNXRUNTIME_INCLUDE_DIR}")

# spacemit_ep（SpaceMIT 推理加速，rv64 apt 安装时自动存在）
find_library(SPACEMIT_EP_LIB
    NAMES spacemit_ep
    HINTS ${_ort_hints}
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH
)

if(SPACEMIT_EP_LIB)
    message(STATUS "SpaceMIT EP:  ${SPACEMIT_EP_LIB}")
else()
    message(STATUS "SpaceMIT EP:  not found (x86_64 build, normal)")
endif()
