/**
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file benchmark_onnx_infer.cpp
 * @brief ONNX Runtime 推理性能基准测试
 *
 * 隔离测试 OnnxRuntimeClass::Run() 的纯推理延迟，排除 obs 组装和动作映射的干扰。
 * 适用场景：测试硬件（K3）的推理引擎性能，或对比不同模型结构（MLP/LSTM/obs_hist）。
 *
 * 用法:
 *   benchmark_onnx_infer <yaml_path> <policy_name> <robot_dir> [--warmup N] [--rounds N] [--verbose]
 *
 * 示例:
 *   benchmark_onnx_infer \
 *     application/native/humanoid_unitree_g1/config/g1.yaml motion \
 *     application/native/humanoid_unitree_g1 \
 *     --warmup 100 --rounds 1000
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "onnx_infer.h"
#include "rl_service.h"

using onnx_runtime::OnnxRuntimeClass;
using rl_policy::LoadedPolicyConfig;
using rl_policy::LoadPolicyConfigFromYaml;

// ---- ANSI 颜色 ----
#define CLR_GREEN "\033[32m"
#define CLR_RED   "\033[31m"
#define CLR_BOLD  "\033[1m"
#define CLR_RESET "\033[0m"

struct Stats {
    double avg;
    double std_dev;
    double p50;
    double p95;
    double p99;
    double max;
};

static Stats compute_stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const int n = static_cast<int>(v.size());
    const double sum = std::accumulate(v.begin(), v.end(), 0.0);
    const double avg = sum / n;
    double sq_sum = 0.0;
    for (double x : v)
        sq_sum += (x - avg) * (x - avg);
    return {avg, std::sqrt(sq_sum / n),
            v[n * 50 / 100],
            v[n * 95 / 100],
            v[n * 99 / 100],
            v.back()};
}

static void parse_args(int argc, char *argv[], int &warmup, int &rounds, int &hz, bool &verbose) {
    warmup  = 100;
    rounds  = 1000;
    hz      = 50;
    verbose = false;
    for (int i = 4; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc)
            warmup = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--rounds") && i + 1 < argc)
            rounds = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--hz") && i + 1 < argc)
            hz = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--verbose"))
            verbose = true;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cerr << "用法: " << argv[0]
                << " <yaml_path> <policy_name> <robot_dir>"
                    " [--warmup N] [--rounds N] [--verbose]\n"
                << "示例:\n  " << argv[0]
                << " application/native/humanoid_unitree_g1/config/g1.yaml"
                    " motion application/native/humanoid_unitree_g1\n";
        return 1;
    }

    int warmup, rounds, hz;
    bool verbose;
    parse_args(argc, argv, warmup, rounds, hz, verbose);

    // 从 YAML 加载策略配置，获取模型路径
    LoadedPolicyConfig loaded_cfg;
    try {
        loaded_cfg = LoadPolicyConfigFromYaml(argv[1], argv[2], argv[3]);
    } catch (const std::exception &e) {
        std::cerr << "[benchmark_onnx_infer] 配置加载失败: " << e.what() << "\n";
        return 1;
    }
    const std::string &model_path = loaded_cfg.exec_cfg.model_path;

    // 加载 ONNX 模型
    OnnxRuntimeClass ort;
    if (!ort.Init(model_path)) {
        std::cerr << "[benchmark_onnx_infer] 模型加载失败: " << model_path << "\n";
        return 1;
    }

    // 填充随机输入（模拟真实推理场景，避免编译器优化掉零输入）
    for (int i = 0; i < ort.GetInputCount(); ++i)
        ort.GetInput(i).setRandom();

    const int64_t obs_dim = ort.GetInputInfo(0).total_size;
    const int64_t act_dim = ort.GetOutputInfo(0).total_size;

    std::cout << CLR_BOLD << std::string(40, '=') << CLR_RESET << "\n"
            << CLR_BOLD << "ONNX Inference Benchmark" << CLR_RESET << "\n"
            << CLR_BOLD << std::string(40, '=') << CLR_RESET << "\n"
            << "Policy:     " << argv[2] << "\n"
            << "Model:      " << model_path << "\n"
            << "Obs dim:    " << obs_dim << "\n"
            << "Action dim: " << act_dim << "\n\n"
            << "Warmup: " << warmup << " rounds\n"
            << "Test:   " << rounds << " rounds\n"
            << std::flush;

    // 预热（消除 ONNX Runtime 图编译、kernel autotune、内存分配等首次开销）
    std::cout << "\n[1/2] 预热中...\n" << std::flush;
    for (int i = 0; i < warmup; ++i)
        ort.Run();

    // 基准测试
    std::cout << "[2/2] 测试中...\n" << std::flush;
    std::vector<double> latencies(rounds);
    for (int i = 0; i < rounds; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ort.Run();
        auto t1 = std::chrono::high_resolution_clock::now();
        latencies[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (verbose)
            std::cout << "  round " << std::setw(4) << i << ": "
                    << std::fixed << std::setprecision(3) << latencies[i] << " ms\n";
    }

    const Stats s = compute_stats(latencies);
    const double deadline_ms = 1000.0 / hz;
    const int miss_count = static_cast<int>(
        std::count_if(latencies.begin(), latencies.end(),
                    [deadline_ms](double x) { return x > deadline_ms; }));
    const double miss_rate = miss_count * 100.0 / rounds;

    std::cout << "\n" << CLR_BOLD << std::string(40, '-') << CLR_RESET << "\n"
            << CLR_BOLD << "Inference Latency (ms)" << CLR_RESET << "\n"
            << CLR_BOLD << std::string(40, '-') << CLR_RESET << "\n"
            << std::fixed << std::setprecision(3)
            << "  Avg:  " << s.avg     << "\n"
            << "  Std:  " << s.std_dev << "\n"
            << "  P50:  " << s.p50     << "\n"
            << "  P95:  " << s.p95     << "\n"
            << "  P99:  " << s.p99     << "\n"
            << "  Max:  " << s.max     << "\n\n";

    std::cout << CLR_BOLD << std::string(40, '-') << CLR_RESET << "\n"
            << CLR_BOLD << "Deadline (" << hz << "Hz = " << std::setprecision(1)
            << deadline_ms << "ms)" << CLR_RESET << "\n"
            << CLR_BOLD << std::string(40, '-') << CLR_RESET << "\n";

    if (miss_count == 0)
        std::cout << CLR_GREEN << CLR_BOLD
                << "  Miss: 0 / " << rounds << " (0.00%)" << CLR_RESET << "\n";
    else
        std::cout << CLR_RED << CLR_BOLD
                << "  Miss: " << miss_count << " / " << rounds
                << " (" << std::setprecision(2) << miss_rate << "%)" << CLR_RESET << "\n";

    std::cout << "\n";
    if (s.max < deadline_ms)
        std::cout << CLR_GREEN << CLR_BOLD
                << "✓ 满足 " << hz << "Hz 实时要求 (Max " << std::setprecision(2)
                << s.max << "ms < " << deadline_ms << "ms)" << CLR_RESET << "\n";
    else
        std::cout << CLR_RED << CLR_BOLD
                << "✗ 不满足 " << hz << "Hz 实时要求 (Max " << std::setprecision(2)
                << s.max << "ms >= " << deadline_ms << "ms)" << CLR_RESET << "\n";

    return 0;
}
