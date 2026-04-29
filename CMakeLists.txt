cmake_minimum_required(VERSION 3.16)
project(rl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# RISC-V 对齐标志（对齐 asr 组件做法）
if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv|RISCV")
    add_compile_options(-mstrict-align -fno-aggressive-loop-optimizations)
endif()

# RPATH：安装后二进制能找到同级 lib/
set(CMAKE_INSTALL_RPATH "\$ORIGIN/../lib:\$ORIGIN")
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)

# ============================================================================
# 依赖
# ============================================================================

find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
include(cmake/FindONNXRuntime.cmake)

option(BUILD_EXAMPLES "Build example programs" ON)

# ============================================================================
# 库：rl
# ============================================================================

add_library(rl SHARED
    src/obs_flat_history.cpp
    src/obs_frame_history.cpp
    src/obs_none.cpp
    src/obs_term.cpp
    src/policy_config_loader.cpp
    src/policy_executor.cpp
    src/backends/onnx_infer.cpp
)

# PUBLIC：调用方 #include "rl_service.h" 时需要 Eigen 头文件
target_include_directories(rl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${EIGEN3_INCLUDE_DIR}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/backends
        ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(rl
    PUBLIC  ${ONNXRUNTIME_LIB}
    PRIVATE yaml-cpp
)

if(SPACEMIT_EP_LIB)
    target_link_libraries(rl PUBLIC ${SPACEMIT_EP_LIB})
endif()

# ============================================================================
# 安装
# ============================================================================

install(TARGETS rl
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES include/rl_service.h
    DESTINATION include
)

# ============================================================================
# 示例
# ============================================================================

if(BUILD_EXAMPLES)
    add_executable(test_policy_executor example/test_policy_executor.cpp)
    target_link_libraries(test_policy_executor PRIVATE rl)
    target_include_directories(test_policy_executor PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    set_target_properties(test_policy_executor PROPERTIES
        BUILD_RPATH_USE_ORIGIN ON
    )
    install(TARGETS test_policy_executor RUNTIME DESTINATION bin)

    add_executable(test_onnx_infer example/test_onnx_infer.cpp)
    target_link_libraries(test_onnx_infer PRIVATE rl)
    target_include_directories(test_onnx_infer PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/backends
    )
    set_target_properties(test_onnx_infer PROPERTIES
        BUILD_RPATH_USE_ORIGIN ON
    )
    install(TARGETS test_onnx_infer RUNTIME DESTINATION bin)

    # 性能基准测试：纯推理（隔离 ONNX Runtime 性能，排除 obs 组装/动作映射干扰）
    add_executable(benchmark_onnx_infer example/benchmark_onnx_infer.cpp)
    target_link_libraries(benchmark_onnx_infer PRIVATE rl)
    target_include_directories(benchmark_onnx_infer PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/backends
    )
    set_target_properties(benchmark_onnx_infer PROPERTIES
        BUILD_RPATH_USE_ORIGIN ON
    )
    install(TARGETS benchmark_onnx_infer RUNTIME DESTINATION bin)

    # 性能基准测试：完整链路（AssembleObs → Infer → MapActionToTargetPos 分段延迟）
    add_executable(benchmark_policy_executor example/benchmark_policy_executor.cpp)
    target_link_libraries(benchmark_policy_executor PRIVATE rl)
    target_include_directories(benchmark_policy_executor PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    set_target_properties(benchmark_policy_executor PROPERTIES
        BUILD_RPATH_USE_ORIGIN ON
    )
    install(TARGETS benchmark_policy_executor RUNTIME DESTINATION bin)

    install(PROGRAMS
        scripts/run_benchmark_onnx.sh
        scripts/run_benchmark_policy.sh
        DESTINATION bin
    )
endif()
