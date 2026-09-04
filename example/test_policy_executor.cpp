/**
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file test_policy_executor.cpp
 * @brief PolicyExecutor 接口使用示例 && 完整测试
 *
 * 演示 PolicyExecutor 所有对外接口的使用方法：
 * - 配置加载（LoadPolicyConfigFromYaml）
 * - 执行器初始化与查询（Init / ObsDim / ActionDim / FeedbackStateCount）
 * - 自定义标量（SetCustomScalar / GetCustomScalar）
 * - 声明式模型 I/O（SetModelInput / GetModelOutput / GetModelOutputTensor）
 * - 观测组装与推理循环（AssembleObs / Infer / MapActionToTargetPos）
 *
 * 用法:
 *   ./test_policy_executor <yaml配置文件路径> <policy_name> <robot_dir>
 *   policy_name 和 robot_dir 均需单独传入，模拟调用方行为。
 *   生产场景中这两个参数由上层调用方解析后传入。
 *
 * 示例（在 spacemit_robot 仓库根目录执行）:
 *   ./output/staging/bin/test_policy_executor \
 *       application/native/humanoid_unitree_g1/config/g1.yaml \
 *       motion \
 *       application/native/humanoid_unitree_g1/
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rl_service.h"

using rl_policy::LoadedPolicyConfig;
using rl_policy::LoadPolicyConfigFromYaml;
using rl_policy::ModelInputSource;
using rl_policy::ModelOutputTarget;
using rl_policy::PolicyExecutor;
using rl_policy::TensorElementType;

namespace {

void Require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error("[test] " + message);
}

void RequireThrows(const std::function<void()> &fn, const std::string &message) {
    try {
        fn();
    } catch (const std::exception &) {
        return;
    }
    throw std::runtime_error("[test] 未正确拒绝: " + message);
}

std::string InputKey(const rl_policy::ModelInputBindingConfig &binding) {
    return binding.key.empty() ? binding.name : binding.key;
}

std::string OutputKey(const rl_policy::ModelOutputBindingConfig &binding) {
    return binding.key.empty() ? binding.name : binding.key;
}

bool SupportsFloatView(TensorElementType type) {
    switch (type) {
    case TensorElementType::FLOAT32:
    case TensorElementType::UINT8:
    case TensorElementType::INT8:
    case TensorElementType::UINT16:
    case TensorElementType::INT16:
    case TensorElementType::INT32:
    case TensorElementType::INT64:
    case TensorElementType::BOOL:
    case TensorElementType::FLOAT16:
    case TensorElementType::FLOAT64:
    case TensorElementType::UINT32:
    case TensorElementType::UINT64:
    case TensorElementType::BFLOAT16:
        return true;
    case TensorElementType::UNDEFINED:
    case TensorElementType::STRING:
    case TensorElementType::COMPLEX64:
    case TensorElementType::COMPLEX128:
    case TensorElementType::FLOAT8E4M3FN:
    case TensorElementType::FLOAT8E4M3FNUZ:
    case TensorElementType::FLOAT8E5M2:
    case TensorElementType::FLOAT8E5M2FNUZ:
    case TensorElementType::UINT4:
    case TensorElementType::INT4:
        return false;
    }
    return false;
}

void ValidateActionClip(
    const std::vector<double> &action, const std::vector<double> &clip_actions) {
    if (clip_actions.empty()) return;
    Require(
        clip_actions.size() == 1 || clip_actions.size() == action.size(),
        "clip_actions 维度错误");
    for (size_t i = 0; i < action.size(); ++i) {
        const double limit = std::abs(
            clip_actions.size() == 1 ? clip_actions.front() : clip_actions[i]);
        Require(
            std::abs(action[i]) <= limit + 1e-6,
            "动作输出超过 clip_actions，index=" + std::to_string(i));
    }
}

void ValidateExposedOutputs(
    const PolicyExecutor &policy, const rl_policy::ModelIOConfig &model_io) {
    for (const auto &binding : model_io.outputs) {
        if (binding.target != ModelOutputTarget::EXPOSE) continue;
        const std::string key = OutputKey(binding);
        const auto tensor = policy.GetModelOutputTensor(key);
        Require(tensor.data != nullptr, "expose 输出为空: " + key);
        Require(tensor.element_count > 0, "expose 输出元素数为 0: " + key);
        Require(tensor.byte_count > 0, "expose 输出字节数为 0: " + key);
        if (SupportsFloatView(tensor.element_type)) {
            const auto &values = policy.GetModelOutput(key);
            Require(
                values.size() == tensor.element_count,
                "expose float 视图维度错误: " + key);
        }
    }
}

}  // namespace

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cerr << "用法: " << argv[0] << " <yaml配置文件> <policy_name> <robot_dir>\n";
        std::cerr << "示例（在 spacemit_robot 仓库根目录执行）:\n";
        std::cerr << "  " << argv[0]
                << " application/native/humanoid_unitree_g1/config/g1.yaml"
                << " motion"
                << " application/native/humanoid_unitree_g1/\n";
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
        //   kp, kd          → 该策略训练时的 PD 增益（每策略独立，进入 RL 后下发给驱动）
        //   infer_decimation, max_roll, max_pitch → behavior_manager 使用

        // ---- 2. 初始化执行器 ----
        std::cout << "[test] 初始化 PolicyExecutor\n";
        PolicyExecutor policy;
        const float invalid_input = 0.0f;
        RequireThrows(
            [&] { policy.SetModelInput("__uninitialized__", &invalid_input, 1); },
            "未初始化时设置 external 输入");
        policy.Init(loaded_cfg.exec_cfg);
        policy.PrintModelInfo();

        // ---- 测试 1: 查询模型属性 ----
        std::cout << "\n========================================\n";
        std::cout << "  测试 1: 模型属性查询接口\n";
        std::cout << "========================================\n";
        std::cout << "  观测维度: " << policy.ObsDim() << "\n";
        std::cout << "  动作维度: " << policy.ActionDim() << "\n";
        std::cout << "  feedback 状态对: " << policy.FeedbackStateCount() << "\n";
        std::cout << "  是否使用 observation_history 输入: "
            << (policy.HasObsHist() ? "是" : "否") << "\n";

        int expected_feedback = 0;
        bool expected_history = false;
        int external_count = 0;
        int constant_count = 0;
        for (const auto &binding : loaded_cfg.exec_cfg.model_io.inputs) {
            expected_feedback += binding.source == ModelInputSource::FEEDBACK ? 1 : 0;
            expected_history = expected_history ||
                binding.source == ModelInputSource::OBSERVATION_HISTORY;
            external_count += binding.source == ModelInputSource::EXTERNAL ? 1 : 0;
            constant_count += binding.source == ModelInputSource::CONSTANT ? 1 : 0;
        }
        const int expose_count = static_cast<int>(std::count_if(
            loaded_cfg.exec_cfg.model_io.outputs.begin(),
            loaded_cfg.exec_cfg.model_io.outputs.end(),
            [](const auto &binding) {
                return binding.target == ModelOutputTarget::EXPOSE;
            }));
        const int ignore_count = static_cast<int>(std::count_if(
            loaded_cfg.exec_cfg.model_io.outputs.begin(),
            loaded_cfg.exec_cfg.model_io.outputs.end(),
            [](const auto &binding) {
                return binding.target == ModelOutputTarget::IGNORE;
            }));

        Require(policy.ObsDim() > 0, "观测维度必须大于 0");
        Require(policy.ActionDim() > 0, "动作维度必须大于 0");
        Require(
            policy.FeedbackStateCount() == expected_feedback,
            "FeedbackStateCount 与 model_io 不一致");
        Require(policy.HasObsHist() == expected_history, "HasObsHist 与 model_io 不一致");
        std::cout << "  model_io: external=" << external_count
                << ", constant=" << constant_count
                << ", expose=" << expose_count
                << ", ignore=" << ignore_count << "\n";

        RequireThrows(
            [&] { policy.SetModelInput("__missing__", &invalid_input, 1); },
            "不存在的 external 输入 key");
        RequireThrows(
            [&] {
                policy.SetModelInput(
                    "__missing__", rl_policy::MakeTensorView(&invalid_input, 1));
            },
            "不存在的原生 external 输入 key");
        RequireThrows(
            [&] { policy.GetModelOutput("__missing__"); },
            "不存在的 expose 输出 key");
        RequireThrows(
            [&] { policy.GetModelOutputTensor("__missing__"); },
            "不存在的原生 expose 输出 key");

        for (const auto &binding : loaded_cfg.exec_cfg.model_io.inputs) {
            if (binding.source != ModelInputSource::EXTERNAL ||
                binding.initial_value.size() <= 1) {
                continue;
            }
            // 单值 default 允许由执行器广播，不能当作完整的运行时 payload 重注入。
            policy.SetModelInput(
                InputKey(binding),
                binding.initial_value.data(),
                static_cast<int>(binding.initial_value.size()));
        }
        for (const auto &binding : loaded_cfg.exec_cfg.model_io.outputs) {
            if (binding.target != ModelOutputTarget::EXPOSE) continue;
            const std::string key = OutputKey(binding);
            RequireThrows(
                [&] { policy.GetModelOutputTensor(key); },
                "首次推理前读取 expose 输出: " + key);
        }

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

        // 循环 5 帧演示推理过程（包括 feedback/history 状态维护）
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
            std::vector<double> raw_action;
            policy.Infer(obs, action, &raw_action);
            if (raw_action.size() != action.size()) {
                throw std::runtime_error("原始动作与实际动作维度不一致");
            }
            ValidateActionClip(action, loaded_cfg.exec_cfg.clip_actions);
            ValidateExposedOutputs(policy, loaded_cfg.exec_cfg.model_io);

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
        ValidateActionClip(action, loaded_cfg.exec_cfg.clip_actions);
        ValidateExposedOutputs(policy, loaded_cfg.exec_cfg.model_io);
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
