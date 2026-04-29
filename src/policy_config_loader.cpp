/**
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file policy_config_loader.cpp
 * @brief 策略 YAML 配置加载实现
 */

#include <yaml-cpp/yaml.h>

#include <filesystem>  // NOLINT(build/c++17)
#include <stdexcept>
#include <string>

#include "rl_service.h"

namespace rl_policy {
namespace {

namespace fs = std::filesystem;

template <typename T>
T NodeAs(const YAML::Node &node, const T &fallback) {
    return node ? node.as<T>() : fallback;
}

}  // namespace

LoadedPolicyConfig LoadPolicyConfigFromYaml(const std::string &yaml_path,
                                            const std::string &policy_name,
                                            const std::string &robot_dir) {
    const fs::path yaml_abs = fs::absolute(yaml_path);
    if (!fs::exists(yaml_abs)) {
        throw std::runtime_error("[PolicyConfigLoader] 配置文件不存在: " + yaml_abs.string());
    }

    YAML::Node cfg = YAML::LoadFile(yaml_abs.string());

    if (!cfg["rl_policy"] || !cfg["rl_policy"]["type"]) {
        throw std::runtime_error("[PolicyConfigLoader] 缺少 rl_policy.type 配置");
    }
    if (cfg["rl_policy"]["type"].as<std::string>() != "onnx_infer") {
        throw std::runtime_error("[PolicyConfigLoader] 不支持的 rl_policy type");
    }

    const auto rl_onnx = cfg["rl_policy"]["onnx_infer"];
    const auto policies = rl_onnx ? rl_onnx["policies"] : YAML::Node{};
    if (!policies || !policies[policy_name]) {
        throw std::runtime_error("[PolicyConfigLoader] 策略 '" + policy_name + "' 的配置不存在");
    }

    const auto policy = policies[policy_name];
    if (!policy["model_path"]) {
        throw std::runtime_error("[PolicyConfigLoader] 策略 '" + policy_name + "' 缺少 model_path");
    }

    LoadedPolicyConfig out;
    const fs::path robot_dir_path(robot_dir);
    out.exec_cfg.model_path =
        fs::weakly_canonical(robot_dir_path / policy["model_path"].as<std::string>()).string();
    if (!fs::exists(out.exec_cfg.model_path)) {
        throw std::runtime_error("[PolicyConfigLoader] ONNX 模型文件不存在: " +
                                out.exec_cfg.model_path);
    }

    if (policy["action_scale"]) {
        if (policy["action_scale"].IsSequence()) {
            out.exec_cfg.action_scale = policy["action_scale"].as<std::vector<double>>();
        } else {
            out.exec_cfg.action_scale = {policy["action_scale"].as<double>()};
        }
    }
    if (policy["action_blend_ratio"]) {
        out.exec_cfg.action_blend_ratio = policy["action_blend_ratio"].as<double>();
    }
    if (policy["action_joint_index"] && policy["action_joint_index"].IsSequence()) {
        out.exec_cfg.action_joint_index = policy["action_joint_index"].as<std::vector<int>>();
    }
    if (policy["rl_default_pos"] && policy["rl_default_pos"].IsSequence()) {
        out.exec_cfg.rl_default_pos = policy["rl_default_pos"].as<std::vector<double>>();
    }
    if (out.exec_cfg.rl_default_pos.empty()) {
        throw std::runtime_error("[PolicyConfigLoader] 策略 '" + policy_name +
                                "' 缺少 rl_default_pos 配置");
    }

    const auto obs = policy["observation"];
    if (obs) {
        out.exec_cfg.ang_vel_scale = NodeAs(obs["ang_vel_scale"], 1.0);
        out.exec_cfg.dof_pos_scale = NodeAs(obs["dof_pos_scale"], 1.0);
        out.exec_cfg.dof_vel_scale = NodeAs(obs["dof_vel_scale"], 1.0);
        out.exec_cfg.euler_angle_scale = NodeAs(obs["euler_angle_scale"], 1.0);
        out.exec_cfg.dof_pos_subtract_default = NodeAs(obs["dof_pos_subtract_default"], true);
        out.exec_cfg.phase_period = NodeAs(obs["phase_period"], 1.0);
        out.exec_cfg.strict_obs_dim_check = NodeAs(obs["strict_dim_check"], true);

        // 读取 flat segments（segment_0_terms / segment_1_terms / ...）
        out.exec_cfg.obs_segments.clear();
        for (int seg_idx = 0;; ++seg_idx) {
            const std::string pfx = "segment_" + std::to_string(seg_idx) + "_";
            if (!obs[pfx + "terms"])
                break;

            ObsSegmentConfig seg;
            for (const auto &t : obs[pfx + "terms"]) {
                seg.terms.push_back(t.as<std::string>());
            }
            seg.mode = NodeAs(obs[pfx + "mode"], std::string(""));
            seg.length = NodeAs(obs[pfx + "length"], 0);
            seg.order = NodeAs(obs[pfx + "order"], std::string("oldest_first"));
            seg.include_current = NodeAs(obs[pfx + "include_current"], true);
            out.exec_cfg.obs_segments.push_back(seg);
        }

        // 读取 custom scalars（parallel arrays）
        const auto scalar_keys = obs["custom_scalar_keys"];
        const auto scalar_vals = obs["custom_scalar_values"];
        if (scalar_keys && scalar_vals && scalar_keys.IsSequence() && scalar_vals.IsSequence()) {
            const auto keys = scalar_keys.as<std::vector<std::string>>();
            const auto vals = scalar_vals.as<std::vector<double>>();
            for (size_t i = 0; i < keys.size() && i < vals.size(); ++i) {
                out.exec_cfg.custom_scalar_defaults[keys[i]] = static_cast<float>(vals[i]);
            }
        }
    }

    if (out.exec_cfg.obs_segments.empty()) {
        throw std::runtime_error("[PolicyConfigLoader] 策略 '" + policy_name +
                                "' 缺少 observation.segments 配置");
    }

    const auto command = policy["command"];
    if (command) {
        if (command["scale"] && command["scale"].IsSequence() && command["scale"].size() >= 3) {
            out.exec_cfg.command_scale[0] = command["scale"][0].as<double>();
            out.exec_cfg.command_scale[1] = command["scale"][1].as<double>();
            out.exec_cfg.command_scale[2] = command["scale"][2].as<double>();
        }
        if (command["init"] && command["init"].IsSequence() && command["init"].size() >= 3) {
            out.command_init[0] = command["init"][0].as<double>();
            out.command_init[1] = command["init"][1].as<double>();
            out.command_init[2] = command["init"][2].as<double>();
        }
    }

    if (policy["infer_decimation"]) {
        out.infer_decimation = policy["infer_decimation"].as<int>();
    }
    out.max_roll = NodeAs(policy["max_roll"], 1.0);
    out.max_pitch = NodeAs(policy["max_pitch"], 1.0);

    const auto gait = policy["gait"];
    if (gait) {
        if (gait["cycle"])
            out.exec_cfg.gait_cycle = gait["cycle"].as<double>();
        out.exec_cfg.gait_left_offset = NodeAs(gait["left_offset"], 0.0);
        out.exec_cfg.gait_right_offset = NodeAs(gait["right_offset"], 0.5);
        out.exec_cfg.gait_left_ratio = NodeAs(gait["left_ratio"], 0.5);
        out.exec_cfg.gait_right_ratio = NodeAs(gait["right_ratio"], 0.5);
    }

    if (policy["motion_length"]) {
        out.exec_cfg.motion_length = policy["motion_length"].as<double>();
    }

    return out;
}

}  // namespace rl_policy
