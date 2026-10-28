# FindMNN.cmake - Find an installed MNN SDK or build the pinned CPU runtime.
# MNN_ROOT (CMake/environment) overrides system and shared-cache discovery.
# Automatic builds support Linux x86_64 and riscv64; other platforms use MNN_ROOT.

include("${CMAKE_CURRENT_LIST_DIR}/FetchThirdParty.cmake")

set(MNN_ROOT "" CACHE PATH "MNN SDK root containing include/MNN and lib")
set(_mnn_commit "bef71b9756a2c77549eddbe33eb97290e3b16602")

function(_mnn_find_root root out_inc out_lib)
    unset(_mnn_inc CACHE)
    unset(_mnn_lib CACHE)
    find_path(_mnn_inc NAMES MNN/Interpreter.hpp
        PATHS "${root}" PATH_SUFFIXES include NO_DEFAULT_PATH ${ARGN})
    find_library(_mnn_lib NAMES MNN
        PATHS "${root}" PATH_SUFFIXES lib/${CMAKE_LIBRARY_ARCHITECTURE} lib lib64
        NO_DEFAULT_PATH ${ARGN})
    if(_mnn_inc AND _mnn_lib)
        set(${out_inc} "${_mnn_inc}" PARENT_SCOPE)
        set(${out_lib} "${_mnn_lib}" PARENT_SCOPE)
    else()
        set(${out_inc} "" PARENT_SCOPE)
        set(${out_lib} "" PARENT_SCOPE)
    endif()
    unset(_mnn_inc CACHE)
    unset(_mnn_lib CACHE)
endfunction()

# Search headers and library together so different installations are not mixed.
unset(MNN_INCLUDE_DIR CACHE)
unset(MNN_LIB CACHE)
set(_mnn_root "${MNN_ROOT}")
if(NOT _mnn_root)
    set(_mnn_root "$ENV{MNN_ROOT}")
endif()
set(MNN_INCLUDE_DIR "")
set(MNN_LIB "")
if(_mnn_root)
    _mnn_find_root("${_mnn_root}" MNN_INCLUDE_DIR MNN_LIB NO_CMAKE_FIND_ROOT_PATH)
    if(NOT MNN_LIB)
        message(FATAL_ERROR
            "MNN_ROOT='${_mnn_root}' must contain include/MNN/Interpreter.hpp and lib/libMNN.so")
    endif()
elseif(NOT CMAKE_CROSSCOMPILING)
    foreach(_mnn_system_root IN ITEMS /usr/local /usr)
        _mnn_find_root("${_mnn_system_root}" MNN_INCLUDE_DIR MNN_LIB)
        if(MNN_LIB)
            set(_mnn_root "${_mnn_system_root}")
            break()
        endif()
    endforeach()
endif()

if(NOT MNN_LIB)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
       NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|riscv64|riscv)$")
        message(FATAL_ERROR "MNN automatic build supports Linux x86_64/riscv64; set MNN_ROOT for this platform")
    endif()
    set(_mnn_rvv_default OFF)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv")
        set(_mnn_rvv_default ON)
    endif()
    option(MNN_USE_RVV "Enable RVV in the automatically built MNN runtime" ${_mnn_rvv_default})
    set(MNN_RVV_MARCH "rv64gcv_zvfh" CACHE STRING "ISA for MNN RVV kernels")
    set(MNN_BUILD_JOBS 4 CACHE STRING "Parallel jobs for the MNN dependency build")
    if(NOT MNN_BUILD_JOBS MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR "MNN_BUILD_JOBS must be a positive integer")
    endif()

    set(_mnn_options
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DMNN_BUILD_SHARED_LIBS=ON -DMNN_SEP_BUILD=OFF -DMNN_LOW_MEMORY=ON
        -DMNN_BUILD_TOOLS=OFF -DMNN_BUILD_CONVERTER=OFF -DMNN_BUILD_DEMO=OFF
        -DMNN_BUILD_TRAIN=OFF -DMNN_BUILD_BENCHMARK=OFF -DMNN_BUILD_TEST=OFF
        -DMNN_OPENMP=OFF -DMNN_USE_THREAD_POOL=ON
        -DMNN_OPENCL=OFF -DMNN_VULKAN=OFF -DMNN_CUDA=OFF
        "-DMNN_USE_RVV=${MNN_USE_RVV}" "-DMNN_RVV_MARCH=${MNN_RVV_MARCH}"
        -DMNN_RVV_FAST_MATH=OFF -DMNN_RVV_SPACEMIT_IME2=OFF)
    # Propagate the SDK compiler and sysroot; never build a host library for a cross target.
    foreach(_mnn_var IN ITEMS
        CMAKE_C_COMPILER CMAKE_CXX_COMPILER CMAKE_ASM_COMPILER
        CMAKE_C_COMPILER_TARGET CMAKE_CXX_COMPILER_TARGET
        CMAKE_SYSROOT CMAKE_C_FLAGS CMAKE_CXX_FLAGS CMAKE_ASM_FLAGS
        CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
        if(DEFINED ${_mnn_var} AND NOT "${${_mnn_var}}" STREQUAL "")
            list(APPEND _mnn_options "-D${_mnn_var}=${${_mnn_var}}")
        endif()
    endforeach()
    set(_mnn_toolchain_hash "")
    if(CMAKE_TOOLCHAIN_FILE)
        get_filename_component(_mnn_toolchain "${CMAKE_TOOLCHAIN_FILE}" ABSOLUTE
            BASE_DIR "${CMAKE_BINARY_DIR}")
        file(SHA256 "${_mnn_toolchain}" _mnn_toolchain_hash)
        list(APPEND _mnn_options "-DCMAKE_TOOLCHAIN_FILE=${_mnn_toolchain}")
    elseif(CMAKE_CROSSCOMPILING)
        list(APPEND _mnn_options "-DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}"
            "-DCMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}")
        if(NOT CMAKE_C_COMPILER)
            message(FATAL_ERROR "MNN cross build requires CMAKE_TOOLCHAIN_FILE or CMAKE_C_COMPILER")
        endif()
    endif()

    # Source is shared; artifacts are separated by target, compiler and build options.
    string(SHA256 _mnn_build_hash
        "${CMAKE_GENERATOR};${CMAKE_SYSTEM_NAME};${CMAKE_SYSTEM_PROCESSOR};${CMAKE_CXX_COMPILER_ID};${CMAKE_CXX_COMPILER_VERSION};${_mnn_toolchain_hash};${_mnn_options}")
    string(SUBSTRING "${_mnn_build_hash}" 0 16 _mnn_build_hash)
    set(_mnn_build_dir
        "${_FETCH_TP_CACHE_ROOT}/mnn/${_mnn_commit}/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}-${_mnn_build_hash}")
    set(_mnn_root "${_mnn_build_dir}/install")
    _mnn_find_root("${_mnn_root}" MNN_INCLUDE_DIR MNN_LIB NO_CMAKE_FIND_ROOT_PATH)

    if(NOT MNN_LIB)
        fetch_thirdparty(
            NAME "mnn/source-${_mnn_commit}"
            GIT_REPO "https://github.com/alibaba/MNN.git"
            GIT_COMMIT "${_mnn_commit}"
            SUBDIR CMakeLists.txt
            OUT_SOURCE_DIR _mnn_source_file)
        get_filename_component(_mnn_source "${_mnn_source_file}" DIRECTORY)
        message(STATUS "Building MNN in ${_mnn_build_dir}")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -S "${_mnn_source}" -B "${_mnn_build_dir}"
                -G "${CMAKE_GENERATOR}" "-DCMAKE_INSTALL_PREFIX=${_mnn_root}" ${_mnn_options}
            RESULT_VARIABLE _mnn_result)
        if(NOT _mnn_result EQUAL 0)
            message(FATAL_ERROR "MNN configure failed: ${_mnn_result}")
        endif()
        execute_process(
            COMMAND "${CMAKE_COMMAND}" --build "${_mnn_build_dir}"
                --config Release --parallel "${MNN_BUILD_JOBS}"
            RESULT_VARIABLE _mnn_result)
        if(NOT _mnn_result EQUAL 0)
            message(FATAL_ERROR "MNN build failed: ${_mnn_result}")
        endif()
        execute_process(
            COMMAND "${CMAKE_COMMAND}" --install "${_mnn_build_dir}" --config Release
            RESULT_VARIABLE _mnn_result)
        if(NOT _mnn_result EQUAL 0)
            message(FATAL_ERROR "MNN install failed: ${_mnn_result}")
        endif()
        file(COPY "${_mnn_source}/LICENSE.txt" DESTINATION "${_mnn_root}/share/mnn")
        _mnn_find_root("${_mnn_root}" MNN_INCLUDE_DIR MNN_LIB NO_CMAKE_FIND_ROOT_PATH)
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MNN REQUIRED_VARS MNN_INCLUDE_DIR MNN_LIB)
if(NOT MNN_FOUND)
    message(FATAL_ERROR "USE_MNN_BACKEND=ON requires a complete MNN SDK")
endif()
if(EXISTS "${_mnn_root}/share/mnn/LICENSE.txt")
    install(FILES "${_mnn_root}/share/mnn/LICENSE.txt" DESTINATION share/rl/thirdparty/mnn)
endif()
message(STATUS "MNN: ${MNN_LIB}")
message(STATUS "  includes: ${MNN_INCLUDE_DIR}")
