/**
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file benchmark_policy_executor.cpp
 * @brief RL 完整链路端到端性能基准测试
 *
 * 测试完整控制链路的分段延迟：AssembleObs → Infer → MapActionToTargetPos
 * 反映真实运控场景的端到端性能，用于识别性能瓶颈和验证实时性要求。
 *
 * 用法:
 *   benchmark_policy_executor <yaml_path> <policy_name> <robot_dir>
 *                             [--warmup N] [--rounds N] [--verbose]
 *
 * 示例:
 *   benchmark_policy_executor \
 *     application/native/humanoid_unitree_g1/config/g1.yaml motion \
 *     application/native/humanoid_unitree_g1 \
 *     --warmup 100 --rounds 1000
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "rl_service.h"

using rl_policy::LoadedPolicyConfig;
using rl_policy::LoadPolicyConfigFromYaml;
using rl_policy::PolicyExecutor;

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

// 打印单行统计表格行
static void print_row(const std::string &label, const Stats &s, bool bold = false) {
    const char *prefix = bold ? CLR_BOLD : "";
    std::cout << prefix
            << "  " << std::left << std::setw(16) << label
            << std::right << std::fixed << std::setprecision(3)
            << std::setw(8) << s.avg
            << std::setw(8) << s.p50
            << std::setw(8) << s.p95
            << std::setw(8) << s.p99
            << std::setw(8) << s.max
            << (bold ? CLR_RESET : "") << "\n";
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

    // 加载策略配置并初始化执行器
    LoadedPolicyConfig loaded_cfg;
    try {
        loaded_cfg = LoadPolicyConfigFromYaml(argv[1], argv[2], argv[3]);
    } catch (const std::exception &e) {
        std::cerr << "[benchmark_policy_executor] 配置加载失败: " << e.what() << "\n";
        return 1;
    }

    PolicyExecutor policy;
    policy.Init(loaded_cfg.exec_cfg);

    const int num_dof = static_cast<int>(loaded_cfg.exec_cfg.rl_default_pos.size());

    std::cout << CLR_BOLD << std::string(56, '=') << CLR_RESET << "\n"
            << CLR_BOLD << "RL Policy Executor Benchmark" << CLR_RESET << "\n"
            << CLR_BOLD << std::string(56, '=') << CLR_RESET << "\n"
            << "Policy:     " << argv[2] << "\n"
            << "Obs dim:    " << policy.ObsDim() << "\n"
            << "Action dim: " << policy.ActionDim() << "\n"
            << "DOF:        " << num_dof << "\n"
            << "LSTM:       " << (policy.HasLstm() ? "yes" : "no") << "\n"
            << "Obs hist:   " << (policy.HasObsHist() ? "yes" : "no") << "\n\n"
            << "Warmup: " << warmup << " rounds\n"
            << "Test:   " << rounds << " rounds\n"
            << std::flush;

    // 模拟传感器数据（固定值，排除外部随机波动对 timing 的干扰）
    const std::array<double, 3> gyro     = {0.01, -0.02, 0.03};
    const std::array<double, 3> rpy      = {0.0, 0.0, 0.0};
    const std::array<double, 4> base_quat = {1.0, 0.0, 0.0, 0.0};
    const std::array<double, 3> base_vel  = {0.0, 0.0, 0.0};
    const double cmd_vx = loaded_cfg.command_init[0];
    const double cmd_vy = loaded_cfg.command_init[1];
    const double cmd_wz = loaded_cfg.command_init[2];
    const std::vector<double> joint_pos = loaded_cfg.exec_cfg.rl_default_pos;
    const std::vector<double> joint_vel(num_dof, 0.0);
    const float dt = static_cast<float>(
        loaded_cfg.exec_cfg.phase_period > 0 ? 0.02 : 0.02);  // 50Hz 控制周期

    Eigen::VectorXf obs;
    std::vector<double> action;
    std::vector<double> target_pos;

    // 预热
    std::cout << "\n[1/2] 预热中...\n" << std::flush;
    for (int i = 0; i < warmup; ++i) {
        policy.AssembleObs(gyro, rpy, cmd_vx, cmd_vy, cmd_wz,
                            joint_pos, joint_vel, base_quat, base_vel, dt, obs);
        policy.Infer(obs, action);
        policy.MapActionToTargetPos(action, target_pos);
    }

    // 基准测试：分段计时
    std::cout << "[2/2] 测试中...\n" << std::flush;

    std::vector<double> obs_lat(rounds);
    std::vector<double> infer_lat(rounds);
    std::vector<double> map_lat(rounds);
    std::vector<double> e2e_lat(rounds);

    for (int i = 0; i < rounds; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        policy.AssembleObs(gyro, rpy, cmd_vx, cmd_vy, cmd_wz,
                            joint_pos, joint_vel, base_quat, base_vel, dt, obs);
        auto t1 = std::chrono::high_resolution_clock::now();
        policy.Infer(obs, action);
        auto t2 = std::chrono::high_resolution_clock::now();
        policy.MapActionToTargetPos(action, target_pos);
        auto t3 = std::chrono::high_resolution_clock::now();

        obs_lat[i]   = std::chrono::duration<double, std::milli>(t1 - t0).count();
        infer_lat[i] = std::chrono::duration<double, std::milli>(t2 - t1).count();
        map_lat[i]   = std::chrono::duration<double, std::milli>(t3 - t2).count();
        e2e_lat[i]   = std::chrono::duration<double, std::milli>(t3 - t0).count();

        if (verbose)
            std::cout << "  round " << std::setw(4) << i
                    << "  obs=" << std::fixed << std::setprecision(3) << obs_lat[i]
                    << "  infer=" << infer_lat[i]
                    << "  map=" << map_lat[i]
                    << "  e2e=" << e2e_lat[i] << " ms\n";
    }

    const Stats s_obs   = compute_stats(obs_lat);
    const Stats s_infer = compute_stats(infer_lat);
    const Stats s_map   = compute_stats(map_lat);
    const Stats s_e2e   = compute_stats(e2e_lat);
    const double deadline_ms = 1000.0 / hz;
    const int miss_count = static_cast<int>(
        std::count_if(e2e_lat.begin(), e2e_lat.end(),
                    [deadline_ms](double x) { return x > deadline_ms; }));
    const double miss_rate = miss_count * 100.0 / rounds;

    // 打印汇总表格
    std::cout << "\n"
            << CLR_BOLD << std::string(56, '-') << CLR_RESET << "\n"
            << CLR_BOLD
            << "  " << std::left << std::setw(16) << "Component (ms)"
            << std::right
            << std::setw(8) << "Avg"
            << std::setw(8) << "P50"
            << std::setw(8) << "P95"
            << std::setw(8) << "P99"
            << std::setw(8) << "Max"
            << CLR_RESET << "\n"
            << CLR_BOLD << std::string(56, '-') << CLR_RESET << "\n";

    print_row("Obs Assembly",   s_obs);
    print_row("Inference",      s_infer);
    print_row("Action Mapping", s_map);

    std::cout << CLR_BOLD << std::string(56, '-') << CLR_RESET << "\n";
    print_row("End-to-End", s_e2e, true);

    // 超时率
    std::cout << "\n"
            << CLR_BOLD << std::string(56, '-') << CLR_RESET << "\n"
            << CLR_BOLD << "Deadline (" << hz << "Hz = " << std::fixed
            << std::setprecision(1) << deadline_ms << "ms)" << CLR_RESET << "\n"
            << CLR_BOLD << std::string(56, '-') << CLR_RESET << "\n";

    if (miss_count == 0)
        std::cout << CLR_GREEN << CLR_BOLD
                << "  Miss: 0 / " << rounds << " (0.00%)" << CLR_RESET << "\n";
    else
        std::cout << CLR_RED << CLR_BOLD
                << "  Miss: " << miss_count << " / " << rounds
                << " (" << std::setprecision(2) << miss_rate << "%)" << CLR_RESET << "\n";

    std::cout << "\n";
    if (s_e2e.max < deadline_ms)
        std::cout << CLR_GREEN << CLR_BOLD
                << "✓ 满足 " << hz << "Hz 实时要求 (Max " << std::setprecision(2)
                << s_e2e.max << "ms < " << deadline_ms << "ms)" << CLR_RESET << "\n";
    else
        std::cout << CLR_RED << CLR_BOLD
                << "✗ 不满足 " << hz << "Hz 实时要求 (Max " << std::setprecision(2)
                << s_e2e.max << "ms >= " << deadline_ms << "ms)" << CLR_RESET << "\n";

    // 瓶颈分析
    if (s_e2e.avg > 0) {
        const double infer_pct = s_infer.avg / s_e2e.avg * 100.0;
        std::cout << "\n瓶颈分析: 推理占端到端耗时 "
                << std::fixed << std::setprecision(1) << infer_pct << "%\n";
    }

    return 0;
}
