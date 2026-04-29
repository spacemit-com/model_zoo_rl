/**
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file rl_service.h
 * @brief RL 策略执行器公共接口
 *
 * rl_service 是 rl 模块的对外统一接口：
 *   - 模型加载（内部委托推理后端，当前支持 ONNX Runtime）
 *   - 段式观测组装（通过可插拔的 ObsSegmentAssembler 策略）
 *   - 推理执行（MLP / LSTM / obs_hist）
 *   - 动作映射
 *
 * 观测组装基于「段拼接」模型：
 *   每个段（segment）独立指定一组 obs terms 和一种基础组装模式：
 *     - (默认)        无历史，输出当前帧
 *     - flat_history   按变量分组的环形历史缓冲
 *     - frame_history  按帧的滑窗历史缓冲
 *   多段按顺序拼接为完整观测向量。
 */
#ifndef RL_SERVICE_H
#define RL_SERVICE_H

#include <Eigen/Dense>
#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace rl_policy {

// ============================================================
// 观测段配置
// ============================================================

/**
 * @brief 单个观测段的配置
 *
 * mode:
 *   - ""  / 未设置 : 无历史，仅输出当前帧
 *   - "flat_history" : 按变量分组的环形历史
 *   - "frame_history": 按帧的滑窗历史
 *
 * order:
 *   - "oldest_first" : [t-N, ..., t-1]
 *   - "newest_first" : [t-1, ..., t-N]
 *
 * include_current:
 *   - true  : 先写入当前帧再读取历史（RoboMimic 风格）
 *   - false : 先读取历史再写入当前帧（青龙风格）
 */
struct ObsSegmentConfig {
    std::vector<std::string> terms;      ///< 观测项列表
    std::string mode;                    ///< "" | "flat_history" | "frame_history"
    int length = 0;                      ///< 历史帧数（flat_history / frame_history 使用）
    std::string order = "oldest_first";  ///< 历史排序方向
    bool include_current = true;         ///< 是否在读取前先写入当前帧
};

// ============================================================
// 策略执行器配置（YAML 驱动）
// ============================================================

struct PolicyExecutorConfig {
    // ---- 模型与动作 ----
    std::string model_path;
    std::vector<double> action_scale = {0.25};
    double action_blend_ratio = 1.0;
    std::vector<double> rl_default_pos;
    std::vector<int> action_joint_index;

    // ---- 段式观测配置（必须） ----
    std::vector<ObsSegmentConfig> obs_segments;

    // ---- 观测归一化参数 ----
    double ang_vel_scale = 1.0;
    double dof_pos_scale = 1.0;
    double dof_vel_scale = 1.0;
    double euler_angle_scale = 1.0;
    std::array<double, 3> command_scale = {1.0, 1.0, 1.0};
    bool dof_pos_subtract_default = true;

    // ---- phase / gait_phase 参数 ----
    double phase_period = 1.0;
    double gait_cycle = 0.85;
    double gait_left_offset = 0.0;
    double gait_right_offset = 0.5;
    double gait_left_ratio = 0.5;
    double gait_right_ratio = 0.5;

    // ---- ref_motion_phase 参数 ----
    double motion_length = 0.0;

    // ---- 自定义标量默认值 ----
    std::unordered_map<std::string, float> custom_scalar_defaults;

    // ---- 维度校验 ----
    bool strict_obs_dim_check = false;
};

// ============================================================
// 策略 YAML 解析结果
// ============================================================

struct LoadedPolicyConfig {
    PolicyExecutorConfig exec_cfg;
    std::array<double, 3> command_init = {0.0, 0.0, 0.0};
    int infer_decimation = 4;
    double max_roll = 0.7;
    double max_pitch = 0.7;
};

/**
 * @brief 从 YAML 加载指定策略配置
 *
 * 会解析并校验：
 * - 模型路径、动作映射、观测段配置
 * - command.scale / command.init
 * - infer_decimation / max_roll / max_pitch / thread
 *
 * @param yaml_path   YAML 配置文件路径
 * @param policy_name 策略名（对应 rl_policy.onnx_infer.<policy_name>）
 * @param robot_dir   机器人资源根目录（绝对路径），用于解析 model_path 等相对路径
 */
LoadedPolicyConfig LoadPolicyConfigFromYaml(const std::string &yaml_path,
                                            const std::string &policy_name,
                                            const std::string &robot_dir);

// ============================================================
// 策略执行器
// ============================================================

class PolicyExecutor {
public:
    PolicyExecutor();
    ~PolicyExecutor();

    // 禁止拷贝，允许移动
    PolicyExecutor(const PolicyExecutor &) = delete;
    PolicyExecutor &operator=(const PolicyExecutor &) = delete;
    PolicyExecutor(PolicyExecutor &&) noexcept;
    PolicyExecutor &operator=(PolicyExecutor &&) noexcept;

    /**
     * @brief 初始化策略执行器
     * @param cfg 策略配置（模型路径、动作映射、观测段等）
     */
    void Init(const PolicyExecutorConfig &cfg);

    /** @return 观测向量维度 */
    int ObsDim() const;

    /** @return 动作向量维度 */
    int ActionDim() const;

    /** @return 是否为 LSTM 模型 */
    bool HasLstm() const;

    /** @return 是否使用 obs_hist 输入 */
    bool HasObsHist() const;

    /** @brief 打印模型信息 */
    void PrintModelInfo() const;

    /** 设置自定义标量（如 "z", "stand_flag"），需在 AssembleObs 前调用 */
    void SetCustomScalar(const std::string &name, float value);
    float GetCustomScalar(const std::string &name) const;

    /**
     * @brief 组装观测向量
     *
     * 按 segments 配置顺序：计算每段当前帧 → 交给对应 assembler → 拼接输出
     */
    void AssembleObs(const std::array<double, 3> &gyro,
                    const std::array<double, 3> &rpy,
                    double cmd_vx,
                    double cmd_vy,
                    double cmd_wz,
                    const std::vector<double> &joint_pos,
                    const std::vector<double> &joint_vel,
                    const std::array<double, 4> &base_quat,
                    const std::array<double, 3> &base_vel,
                    float dt,
                    Eigen::VectorXf &out_obs);

    /** 执行推理（MLP / LSTM / obs_hist 均自动处理） */
    void Infer(const Eigen::VectorXf &obs, std::vector<double> &out_action);

    /** 将策略动作映射为全身关节目标位置 */
    void MapActionToTargetPos(const std::vector<double> &action,
                            std::vector<double> &target_pos) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rl_policy

#endif  // RL_SERVICE_H
