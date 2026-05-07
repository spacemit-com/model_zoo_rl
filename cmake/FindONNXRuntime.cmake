# FindONNXRuntime.cmake
#
# 查找 ONNX Runtime 头文件和库；x86_64 上找不到时自动拉取 prebuilt release 到 ~/.cache/thirdparty/。
#
# 输入变量（可选）:
#   ONNXRUNTIME_DIR  — ONNX Runtime 安装根目录（CMake 变量或同名环境变量）
#
# 输出变量:
#   ONNXRUNTIME_INCLUDE_DIR  — 头文件目录（含 onnxruntime_cxx_api.h）
#   ONNXRUNTIME_LIB          — libonnxruntime 路径
#   SPACEMIT_EP_LIB          — libspacemit_ep 路径（rv64 apt 安装时自动找到，否则为空）
#
# 查找优先级（先检测、再 fetch 兜底）:
#   1. -DONNXRUNTIME_DIR=...
#   2. 环境变量 ONNXRUNTIME_DIR
#   3. /usr/local（x86_64 用户手动安装）
#   4. /usr（rv64 apt 装的 spacemit-onnxruntime）
#   5. ~/.cache/thirdparty/onnxruntime/onnxruntime-linux-x64-1.21.0/（之前 fetch 留下的）
#   6. 兜底：仅 x86_64 触发 fetch_thirdparty 拉取并解压到 cache，再次 find

set(_ORT_VERSION "1.21.0")
set(_ORT_X64_RELEASE "onnxruntime-linux-x64-${_ORT_VERSION}")
set(_ORT_X64_URL "https://github.com/microsoft/onnxruntime/releases/download/v${_ORT_VERSION}/${_ORT_X64_RELEASE}.tgz")

function(_ort_find_in_hints out_inc out_lib)
    find_path(_ort_inc
        NAMES onnxruntime_cxx_api.h
        HINTS ${ARGN}
        PATH_SUFFIXES
            include/onnxruntime/core/session
            include/onnxruntime
            include
        NO_DEFAULT_PATH
    )
    find_library(_ort_lib
        NAMES onnxruntime
        HINTS ${ARGN}
        PATH_SUFFIXES lib
        NO_DEFAULT_PATH
    )
    set(${out_inc} "${_ort_inc}" PARENT_SCOPE)
    set(${out_lib} "${_ort_lib}" PARENT_SCOPE)
endfunction()

# ---- 步骤 1：组装 hints（含 cache 路径） ----
set(_ort_hints "")
if(DEFINED ONNXRUNTIME_DIR)
    list(APPEND _ort_hints "${ONNXRUNTIME_DIR}")
endif()
if(DEFINED ENV{ONNXRUNTIME_DIR})
    list(APPEND _ort_hints "$ENV{ONNXRUNTIME_DIR}")
endif()
list(APPEND _ort_hints /usr/local /usr)
if(DEFINED ENV{HOME})
    list(APPEND _ort_hints "$ENV{HOME}/.cache/thirdparty/onnxruntime/${_ORT_X64_RELEASE}")
endif()

# ---- 步骤 2：先 find 一次（命中预装 / 已 fetch 的 cache） ----
_ort_find_in_hints(ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIB ${_ort_hints})

# ---- 步骤 3：找不到 + x86_64 → 触发自动 fetch ----
if((NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
   AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    include("${CMAKE_CURRENT_LIST_DIR}/FetchThirdParty.cmake")
    fetch_thirdparty(
        NAME onnxruntime
        ARCHIVE_URL "${_ORT_X64_URL}"
        ARCHIVE_SUBDIR "${_ORT_X64_RELEASE}"
        OUT_SOURCE_DIR _ort_fetched_dir
    )
    # 再 find 一次（cache 路径优先）
    unset(ONNXRUNTIME_INCLUDE_DIR CACHE)
    unset(ONNXRUNTIME_LIB CACHE)
    _ort_find_in_hints(ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIB "${_ort_fetched_dir}" ${_ort_hints})
endif()

# ---- 步骤 4：仍找不到 → fatal ----
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR
        "ONNX Runtime not found.\n"
        "  x86_64: 网络不通时手动下载 ${_ORT_X64_URL}\n"
        "          解压后 export ONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-${_ORT_VERSION}\n"
        "  rv64:   sudo apt install spacemit-onnxruntime\n"
        "  或编译时显式指定: cmake .. -DONNXRUNTIME_DIR=/path/to/onnxruntime")
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
