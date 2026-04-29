/**
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file obs_term.cpp
 * @brief 观测项计算实现
 */

#include "obs_term.h"

#include <algorithm>
#include <string>
#include <vector>
#include <cmath>

namespace rl_policy {

void ObsTermCalculator::Init(const ObsTermConfig &config,
                            const std::vector<double> &default_pos,
                            const std::vector<int> &action_joint_index,
                            int action_dim) {
    cfg_ = config;
    default_pos_ = default_pos;
    joint_index_ = action_joint_index;
    action_dim_ = action_dim;
    phase_time_ = 0.0f;
}

int ObsTermCalculator::JointCount() const {
    if (!joint_index_.empty())
        return static_cast<int>(joint_index_.size());
    return static_cast<int>(default_pos_.size());
}

int ObsTermCalculator::JointAt(int i) const {
    if (!joint_index_.empty())
        return joint_index_[i];
    return i;
}

// ============================================================
// TermDim
// ============================================================

int ObsTermCalculator::TermDim(const std::string &term) const {
    if (term == "ang_vel")
        return 3;
    if (term == "base_lin_vel")
        return 3;
    if (term == "gravity" || term == "projected_gravity")
        return 3;
    if (term == "euler_angle" || term == "rpy" || term == "quat")
        return 3;
    if (term == "command")
        return 3;
    if (term == "dof_pos")
        return JointCount();
    if (term == "dof_vel")
        return JointCount();
    if (term == "last_action")
        return action_dim_;
    if (term == "phase")
        return 2;
    if (term == "gait_clock")
        return 2;
    if (term == "gait_phase")
        return 6;
    if (term == "ref_motion_phase")
        return 1;
    // 自定义标量
    return 1;
}

// ============================================================
// FillTermValues
// ============================================================

void ObsTermCalculator::FillTermValues(const std::string &term,
                                        const std::array<double, 3> &gyro,
                                        const std::array<double, 3> &rpy,
                                        double cmd_vx,
                                        double cmd_vy,
                                        double cmd_wz,
                                        const std::vector<double> &joint_pos,
                                        const std::vector<double> &joint_vel,
                                        const std::vector<double> &last_action,
                                        const std::array<double, 4> &base_quat,
                                        const std::array<double, 3> &base_vel,
                                        float *out) const {
    if (term == "ang_vel") {
        out[0] = static_cast<float>(gyro[0] * cfg_.ang_vel_scale);
        out[1] = static_cast<float>(gyro[1] * cfg_.ang_vel_scale);
        out[2] = static_cast<float>(gyro[2] * cfg_.ang_vel_scale);
        return;
    }

    if (term == "base_lin_vel") {
        // 世界系线速度 → 机体系（四元数共轭旋转）
        double w = base_quat[0], x = -base_quat[1], y = -base_quat[2], z = -base_quat[3];
        double vx = base_vel[0], vy = base_vel[1], vz = base_vel[2];
        double cx = y * vz - z * vy + w * vx;
        double cy = z * vx - x * vz + w * vy;
        double cz = x * vy - y * vx + w * vz;
        out[0] = static_cast<float>(vx + 2.0 * (y * cz - z * cy));
        out[1] = static_cast<float>(vy + 2.0 * (z * cx - x * cz));
        out[2] = static_cast<float>(vz + 2.0 * (x * cy - y * cx));
        return;
    }

    if (term == "gravity" || term == "projected_gravity") {
        const double r = rpy[0], p = rpy[1];
        out[0] = static_cast<float>(std::sin(p));
        out[1] = static_cast<float>(-std::sin(r) * std::cos(p));
        out[2] = static_cast<float>(-std::cos(r) * std::cos(p));
        return;
    }

    if (term == "euler_angle" || term == "rpy" || term == "quat") {
        out[0] = static_cast<float>(rpy[0] * cfg_.euler_angle_scale);
        out[1] = static_cast<float>(rpy[1] * cfg_.euler_angle_scale);
        out[2] = static_cast<float>(rpy[2] * cfg_.euler_angle_scale);
        return;
    }

    if (term == "command") {
        out[0] = static_cast<float>(cmd_vx * cfg_.command_scale[0]);
        out[1] = static_cast<float>(cmd_vy * cfg_.command_scale[1]);
        out[2] = static_cast<float>(cmd_wz * cfg_.command_scale[2]);
        return;
    }

    if (term == "dof_pos") {
        const int n = JointCount();
        for (int i = 0; i < n; ++i) {
            const int j = JointAt(i);
            float value = 0.0f;
            if (j >= 0 && j < static_cast<int>(joint_pos.size())) {
                if (cfg_.dof_pos_subtract_default && j < static_cast<int>(default_pos_.size())) {
                    value =
                        static_cast<float>((joint_pos[j] - default_pos_[j]) * cfg_.dof_pos_scale);
                } else {
                    value = static_cast<float>(joint_pos[j] * cfg_.dof_pos_scale);
                }
            }
            out[i] = value;
        }
        return;
    }

    if (term == "dof_vel") {
        const int n = JointCount();
        for (int i = 0; i < n; ++i) {
            const int j = JointAt(i);
            float value = 0.0f;
            if (j >= 0 && j < static_cast<int>(joint_vel.size())) {
                value = static_cast<float>(joint_vel[j] * cfg_.dof_vel_scale);
            }
            out[i] = value;
        }
        return;
    }

    if (term == "last_action") {
        for (int i = 0; i < action_dim_; ++i) {
            out[i] = (i < static_cast<int>(last_action.size())) ? static_cast<float>(last_action[i])
                                                                : 0.0f;
        }
        return;
    }

    if (term == "phase") {
        const double period = (cfg_.phase_period > 1e-6) ? cfg_.phase_period : 0.8;
        double ph = std::fmod(static_cast<double>(phase_time_) / period, 1.0);
        if (ph < 0.0)
            ph += 1.0;
        constexpr float kTwoPi = 6.28318530717958647692f;
        const float pr = kTwoPi * static_cast<float>(ph);
        out[0] = std::sin(pr);
        out[1] = std::cos(pr);
        return;
    }

    if (term == "gait_clock") {
        const double period = (cfg_.phase_period > 1e-6) ? cfg_.phase_period : 0.8;
        double ph = std::fmod(static_cast<double>(phase_time_) / period, 1.0);
        if (ph < 0.0)
            ph += 1.0;
        constexpr float kTwoPi = 6.28318530717958647692f;
        const float pr = kTwoPi * static_cast<float>(ph);
        out[0] = std::cos(pr);
        out[1] = std::sin(pr);
        return;
    }

    if (term == "gait_phase") {
        const double gc = (cfg_.gait_cycle > 1e-6) ? cfg_.gait_cycle : 0.85;
        const double t = static_cast<double>(phase_time_);
        double lp = std::fmod(t / gc + cfg_.gait_left_offset, 1.0);
        double rp_val = std::fmod(t / gc + cfg_.gait_right_offset, 1.0);
        if (lp < 0.0)
            lp += 1.0;
        if (rp_val < 0.0)
            rp_val += 1.0;
        constexpr float kTwoPi = 6.28318530717958647692f;
        out[0] = std::sin(kTwoPi * static_cast<float>(lp));
        out[1] = std::sin(kTwoPi * static_cast<float>(rp_val));
        out[2] = std::cos(kTwoPi * static_cast<float>(lp));
        out[3] = std::cos(kTwoPi * static_cast<float>(rp_val));
        out[4] = static_cast<float>(cfg_.gait_left_ratio);
        out[5] = static_cast<float>(cfg_.gait_right_ratio);
        return;
    }

    if (term == "ref_motion_phase") {
        double ml = cfg_.motion_length;
        if (ml <= 0.0)
            ml = 1.0;
        double ref = std::min(static_cast<double>(phase_time_), ml) / ml;
        out[0] = static_cast<float>(std::min(ref, 1.0));
        return;
    }

    // 自定义标量
    auto it = custom_scalars_.find(term);
    out[0] = (it != custom_scalars_.end()) ? it->second : 0.0f;
}

// ============================================================
// 自定义标量
// ============================================================

void ObsTermCalculator::SetCustomScalar(const std::string &name, float value) {
    custom_scalars_[name] = value;
}

float ObsTermCalculator::GetCustomScalar(const std::string &name) const {
    auto it = custom_scalars_.find(name);
    return (it != custom_scalars_.end()) ? it->second : 0.0f;
}

void ObsTermCalculator::AdvancePhase(float dt) {
    phase_time_ += dt;
}

void ObsTermCalculator::ResetPhase() {
    phase_time_ = 0.0f;
}

}  // namespace rl_policy
