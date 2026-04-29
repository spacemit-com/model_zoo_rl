/**
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file obs_term.h
 * @brief 观测项计算器
 *
 * 统一管理所有 obs term 的维度查询和值计算。
 * 支持的内建 term:
 *   ang_vel(3), gravity/projected_gravity(3), euler_angle/rpy/quat(3),
 *   command(3), dof_pos(N), dof_vel(N), last_action(M),
 *   phase(2), gait_phase(6), ref_motion_phase(1)
 * 未知 term 视为自定义标量（dim=1），通过 custom_scalars 查值。
 */

#ifndef OBS_TERM_H
#define OBS_TERM_H


#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace rl_policy {

/// 观测归一化参数
struct ObsTermConfig {
    double ang_vel_scale = 1.0;
    double dof_pos_scale = 1.0;
    double dof_vel_scale = 1.0;
    double euler_angle_scale = 1.0;
    std::array<double, 3> command_scale = {1.0, 1.0, 1.0};
    bool dof_pos_subtract_default = true;
    double phase_period = 1.0;
    double gait_cycle = 0.85;
    double gait_left_offset = 0.0;
    double gait_right_offset = 0.5;
    double gait_left_ratio = 0.5;
    double gait_right_ratio = 0.5;
    double motion_length = 0.0;
};

class ObsTermCalculator {
public:
    void Init(const ObsTermConfig &config,
            const std::vector<double> &default_pos,
            const std::vector<int> &action_joint_index,
            int action_dim);

    /// 查询某 term 的维度
    int TermDim(const std::string &term) const;

    /// 填充 term 值到 out 缓冲区
    void FillTermValues(const std::string &term,
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
                        float *out) const;

    void SetCustomScalar(const std::string &name, float value);
    float GetCustomScalar(const std::string &name) const;

    /// 推进相位时间
    void AdvancePhase(float dt);

    /// 重置相位时间
    void ResetPhase();

private:
    int JointCount() const;
    int JointAt(int i) const;

    ObsTermConfig cfg_;
    std::vector<double> default_pos_;
    std::vector<int> joint_index_;
    int action_dim_ = 0;
    float phase_time_ = 0.0f;
    std::unordered_map<std::string, float> custom_scalars_;
};

}  // namespace rl_policy

#endif  // OBS_TERM_H
