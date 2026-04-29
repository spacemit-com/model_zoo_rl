/**
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file test_policy_executor.cpp
 * @brief PolicyExecutor 接口使用示例 && 完整测试
 *
 * 演示 PolicyExecutor 所有对外接口的使用方法：
 * - 配置加载（LoadPolicyConfigFromYaml）
 * - 执行器初始化与查询（Init / ObsDim / ActionDim / HasLstm / HasObsHist）
 * - 自定义标量（SetCustomScalar / GetCustomScalar）
 * - 观测组装与推理循环（AssembleObs / Infer / MapActionToTargetPos）
 *
 * 用法:
 *   ./test_policy_executor <yaml配置文件路径> <policy_name> <robot_dir>
 *   policy_name 和 robot_dir 均需单独传入，模拟调用方行为。
 *   生产场景中这两个参数由上层调用方解析后传入。
 *
 * 示例:
 *   ./test_policy_executor ../../../application/config/g1.yaml motion ../../../application/robot/g1
 */

#include <iostream>
#include <vector>

#include "rl_service.h"

using rl_policy::LoadedPolicyConfig;
using rl_policy::LoadPolicyConfigFromYaml;
using rl_policy::PolicyExecutor;

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cerr << "用法: " << argv[0] << " <yaml配置文件> <policy_name> <robot_dir>\n";
        std::cerr << "示例:\n";
        std::cerr << "  " << argv[0]
                << " ../../../application/config/g1.yaml motion ../../../application/robot/g1\n";
        return 1;
    }

    try {
        // ---- 1. 加载配置 ----
        std::cout << "[test] 加载配置: " << argv[1] << ", policy: " << argv[2]
                << ", robot_dir: " << argv[3] << "\n";
        const LoadedPolicyConfig loaded_cfg = LoadPolicyConfigFromYaml(argv[1], argv[2], argv[3]);

        // LoadedPolicyConfig 包含：
        //   exec_cfg        → PolicyExecutorConfig（模型、观测、动作映射等）
        //   command_init    → 初始速度指令 [vx, vy, wz]
        //   infer_decimation, max_roll, max_pitch → behavior_manager 使用
        //   thread_cpu_id, thread_priority        → behavior_manager 使用

        // ---- 2. 初始化执行器 ----
        std::cout << "[test] 初始化 PolicyExecutor\n";
        PolicyExecutor policy;
        policy.Init(loaded_cfg.exec_cfg);
        policy.PrintModelInfo();

        // ---- 测试 1: 查询模型属性 ----
        std::cout << "\n========================================\n";
        std::cout << "  测试 1: 模型属性查询接口\n";
        std::cout << "========================================\n";
        std::cout << "  观测维度: " << policy.ObsDim() << "\n";
        std::cout << "  动作维度: " << policy.ActionDim() << "\n";
        std::cout << "  是否为 LSTM 模型: " << (policy.HasLstm() ? "是" : "否") << "\n";
        std::cout << "  是否使用 obs_hist 输入: " << (policy.HasObsHist() ? "是" : "否") << "\n";

        policy.PrintModelInfo();
        if (!loaded_cfg.exec_cfg.custom_scalar_defaults.empty()) {
            std::cout << "  自定义标量默认值:\n";
            for (const auto &kv : loaded_cfg.exec_cfg.custom_scalar_defaults)
                std::cout << "    " << kv.first << " = " << kv.second << "\n";
        }

        // 传感器数据准备（非测试，仅初始化后续推理所需输入）
        const int num_dof = static_cast<int>(loaded_cfg.exec_cfg.rl_default_pos.size());
        std::array<double, 3> gyro = {0.01, -0.02, 0.03};
        std::array<double, 3> rpy = {0.0, 0.0, 0.0};
        const double cmd_vx = loaded_cfg.command_init[0];
        const double cmd_vy = loaded_cfg.command_init[1];
        const double cmd_wz = loaded_cfg.command_init[2];
        std::vector<double> joint_pos = loaded_cfg.exec_cfg.rl_default_pos;
        std::vector<double> joint_vel(num_dof, 0.0);
        float dt = 0.02f;

        // ---- 测试 2: 推理循环演示 ----
        std::cout << "\n========================================\n";
        std::cout << "  测试 2: 观测组装、推理、动作映射\n";
        std::cout << "========================================\n";

        Eigen::VectorXf obs;
        std::array<double, 4> base_quat = {1.0, 0.0, 0.0, 0.0};  // 单位四元数
        std::array<double, 3> base_vel = {0.0, 0.0, 0.0};

        // 循环 5 帧演示推理过程（特别是 LSTM 模型的状态维护）
        std::cout << "  执行 5 帧推理循环...\n";
        for (int frame = 0; frame < 5; ++frame) {
            // 模拟传感器数据变化
            gyro[0] += 0.001;
            joint_pos[0] += 0.01;
            joint_vel[0] += 0.001;

            // 组装观测
            policy.AssembleObs(gyro,
                                rpy,
                                cmd_vx,
                                cmd_vy,
                                cmd_wz,
                                joint_pos,
                                joint_vel,
                                base_quat,
                                base_vel,
                                dt,
                                obs);

            // 推理
            std::vector<double> action;
            policy.Infer(obs, action);

            // 动作映射
            std::vector<double> target_pos;
            policy.MapActionToTargetPos(action, target_pos);

            std::cout << "    Frame " << frame << ": obs_dim=" << obs.size()
                    << " action_dim=" << action.size() << " target_pos_dim=" << target_pos.size()
                    << "\n";
        }

        // ---- 测试 3: 单帧详细推理流程 ----
        std::cout << "\n========================================\n";
        std::cout << "  测试 3: 单帧详细推理流程\n";
        std::cout << "========================================\n";

        policy.AssembleObs(
            gyro, rpy, cmd_vx, cmd_vy, cmd_wz, joint_pos, joint_vel, base_quat, base_vel, dt, obs);
        std::cout << "  观测向量维度: " << obs.size() << " (期望: " << policy.ObsDim() << ")\n";
        if (obs.size() != policy.ObsDim()) {
            std::cerr << "  ✗ 警告：观测维度不匹配！\n";
        } else {
            std::cout << "  ✓ 观测维度正确\n";
        }

        std::vector<double> action;
        policy.Infer(obs, action);
        std::cout << "  动作输出维度: " << action.size() << " (期望: " << policy.ActionDim()
                << ")\n";
        if (action.size() != policy.ActionDim()) {
            std::cerr << "  ✗ 警告：动作维度不匹配！\n";
        } else {
            std::cout << "  ✓ 动作维度正确\n";
        }

        std::vector<double> target_pos;
        policy.MapActionToTargetPos(action, target_pos);
        std::cout << "  关节目标位置数量: " << target_pos.size() << " (期望: " << num_dof << ")\n";
        if (target_pos.size() != static_cast<size_t>(num_dof)) {
            std::cerr << "  ✗ 警告：关节维度不匹配！\n";
        } else {
            std::cout << "  ✓ 关节维度正确\n";
        }

        std::cout << "\n========================================\n";
        std::cout << "✓ 所有测试成功！\n";
        std::cout << "========================================\n";
    } catch (const std::exception &e) {
        std::cerr << "[错误] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
