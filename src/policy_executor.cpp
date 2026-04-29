/**
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file policy_executor.cpp
 * @brief 策略执行器实现（编排层）
 *
 * 职责：
 *   1. 加载 ONNX 模型（通过内部 onnx_infer 后端）
 *   2. 为每段创建对应的 ObsSegmentAssembler
 *   3. 每步：计算 term 值 → 交给 assembler → 拼接 → 推理
 *   4. 动作映射到关节目标位置
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "obs_assembler.h"
#include "obs_flat_history.h"
#include "obs_frame_history.h"
#include "obs_none.h"
#include "obs_term.h"
#include "onnx_infer.h"
#include "rl_service.h"

namespace rl_policy {

// ============================================================
// 工厂函数：根据 mode 创建对应的 assembler
// ============================================================

static std::unique_ptr<ObsSegmentAssembler> CreateAssembler(const std::string &mode) {
    if (mode.empty()) {
        return std::make_unique<ObsNone>();
    }
    if (mode == "flat_history") {
        return std::make_unique<ObsFlatHistory>();
    }
    if (mode == "frame_history") {
        return std::make_unique<ObsFrameHistory>();
    }
    throw std::runtime_error("[PolicyExecutor] 未知段模式: " + mode);
}

// ============================================================
// Impl（PIMPL）
// ============================================================

class PolicyExecutor::Impl {
public:
    PolicyExecutorConfig cfg;
    onnx_runtime::OnnxRuntimeClass onnx;
    ObsTermCalculator term_calc;

    struct SegmentRuntime {
        ObsSegmentConfig seg_cfg;
        std::vector<TermLayout> term_layouts;
        int frame_dim = 0;
        std::unique_ptr<ObsSegmentAssembler> assembler;
    };

    std::vector<SegmentRuntime> segments;

    bool initialized = false;
    bool has_lstm = false;
    bool has_obs_hist = false;
    int obs_dim = 0;
    int action_dim = 0;
    int expected_obs_dim = 0;
    int obs_hist_len = 0;

    // LSTM 状态
    Eigen::VectorXf h_state;
    Eigen::VectorXf c_state;

    // 历史观测窗口（obs_hist 输入张量的模型）
    std::vector<Eigen::VectorXf> obs_hist;

    // 内部维护的平滑动作，用于 last_action 与输出一致
    std::vector<double> blended_action;
};

namespace {

std::array<double, 3> NormalizeRpy(const std::array<double, 3> &rpy) {
    std::array<double, 3> out = rpy;
    constexpr double kTwoPi = 2.0 * M_PI;
    for (double &v : out) {
        while (v > M_PI)
            v -= kTwoPi;
        while (v < -M_PI)
            v += kTwoPi;
    }
    return out;
}

double ClampBlendRatio(double ratio) {
    return std::clamp(ratio, 0.0, 1.0);
}

}  // namespace

// ============================================================
// PolicyExecutor 公共接口
// ============================================================

PolicyExecutor::PolicyExecutor() : impl_(std::make_unique<Impl>()) {}
PolicyExecutor::~PolicyExecutor() = default;
PolicyExecutor::PolicyExecutor(PolicyExecutor &&) noexcept = default;
PolicyExecutor &PolicyExecutor::operator=(PolicyExecutor &&) noexcept = default;

// ---- 查询 ----

int PolicyExecutor::ObsDim() const {
    return impl_->obs_dim;
}
int PolicyExecutor::ActionDim() const {
    return impl_->action_dim;
}
bool PolicyExecutor::HasLstm() const {
    return impl_->has_lstm;
}
bool PolicyExecutor::HasObsHist() const {
    return impl_->has_obs_hist;
}
void PolicyExecutor::PrintModelInfo() const {
    impl_->onnx.PrintModelInfo();
}

// ---- 自定义标量 ----

void PolicyExecutor::SetCustomScalar(const std::string &name, float value) {
    impl_->term_calc.SetCustomScalar(name, value);
}
float PolicyExecutor::GetCustomScalar(const std::string &name) const {
    return impl_->term_calc.GetCustomScalar(name);
}

// ============================================================
// init
// ============================================================

void PolicyExecutor::Init(const PolicyExecutorConfig &cfg) {
    impl_->cfg = cfg;
    impl_->cfg.action_blend_ratio = ClampBlendRatio(impl_->cfg.action_blend_ratio);
    impl_->segments.clear();
    impl_->term_calc.ResetPhase();

    // ---- 校验：必须提供 segments ----
    if (cfg.obs_segments.empty()) {
        throw std::runtime_error(
            "[PolicyExecutor] 缺少 observation.segments 配置，"
            "请在 YAML 中为每个策略定义段式观测");
    }

    // ---- 模型加载 ----
    if (!impl_->onnx.Init(cfg.model_path)) {
        throw std::runtime_error("[PolicyExecutor] 模型初始化失败: " + cfg.model_path);
    }
    impl_->onnx.PrintModelInfo();

    const int input_count = impl_->onnx.GetInputCount();
    const int output_count = impl_->onnx.GetOutputCount();

    if (input_count == 1 && output_count == 1) {
        impl_->has_lstm = false;
        impl_->has_obs_hist = false;
        impl_->obs_dim = static_cast<int>(impl_->onnx.GetInputInfo(0).total_size);
        impl_->action_dim = static_cast<int>(impl_->onnx.GetOutputInfo(0).total_size);
        std::cout << "[PolicyExecutor] MLP 模型: obs_dim=" << impl_->obs_dim
                << ", action_dim=" << impl_->action_dim << std::endl;

    } else if (input_count == 2 && output_count == 1) {
        impl_->has_lstm = false;
        impl_->has_obs_hist = true;
        impl_->obs_dim = static_cast<int>(impl_->onnx.GetInputInfo(0).total_size);
        impl_->action_dim = static_cast<int>(impl_->onnx.GetOutputInfo(0).total_size);
        const auto &hist_info = impl_->onnx.GetInputInfo(1);
        if (hist_info.shape.size() == 3 && hist_info.shape[0] == 1) {
            impl_->obs_hist_len = static_cast<int>(hist_info.shape[1]);
            const int hist_obs_dim = static_cast<int>(hist_info.shape[2]);
            if (hist_obs_dim != impl_->obs_dim) {
                throw std::runtime_error("[PolicyExecutor] 历史观测维度不匹配: obs_dim=" +
                                        std::to_string(impl_->obs_dim) +
                                        ", obs_hist[2]=" + std::to_string(hist_obs_dim));
            }
        } else {
            throw std::runtime_error("[PolicyExecutor] 历史观测输入形状错误");
        }
        impl_->obs_hist.clear();
        impl_->obs_hist.resize(impl_->obs_hist_len, Eigen::VectorXf::Zero(impl_->obs_dim));
        std::cout << "[PolicyExecutor] 带历史观测模型: obs_dim=" << impl_->obs_dim
                << ", action_dim=" << impl_->action_dim
                << ", obs_hist_len=" << impl_->obs_hist_len << std::endl;

    } else if (input_count == 3 && output_count == 3) {
        impl_->has_lstm = true;
        impl_->has_obs_hist = false;
        impl_->obs_dim = static_cast<int>(impl_->onnx.GetInputInfo(0).total_size);
        impl_->action_dim = static_cast<int>(impl_->onnx.GetOutputInfo(0).total_size);
        const int h_dim = static_cast<int>(impl_->onnx.GetInputInfo(1).total_size);
        const int c_dim = static_cast<int>(impl_->onnx.GetInputInfo(2).total_size);
        impl_->h_state.setZero(h_dim);
        impl_->c_state.setZero(c_dim);
        std::cout << "[PolicyExecutor] LSTM 模型: obs_dim=" << impl_->obs_dim
                << ", action_dim=" << impl_->action_dim << ", h_dim=" << h_dim
                << ", c_dim=" << c_dim << std::endl;

    } else {
        throw std::runtime_error(
            "[PolicyExecutor] 不支持的模型类型: " + std::to_string(input_count) + " 输入, " +
            std::to_string(output_count) + " 输出");
    }

    // 所有模型类型都需要初始化内部动作缓冲
    impl_->blended_action.assign(impl_->action_dim, 0.0);

    // ---- 初始化 term 计算器 ----
    ObsTermConfig tc;
    tc.ang_vel_scale = cfg.ang_vel_scale;
    tc.dof_pos_scale = cfg.dof_pos_scale;
    tc.dof_vel_scale = cfg.dof_vel_scale;
    tc.euler_angle_scale = cfg.euler_angle_scale;
    tc.command_scale = cfg.command_scale;
    tc.dof_pos_subtract_default = cfg.dof_pos_subtract_default;
    tc.phase_period = cfg.phase_period;
    tc.gait_cycle = cfg.gait_cycle;
    tc.gait_left_offset = cfg.gait_left_offset;
    tc.gait_right_offset = cfg.gait_right_offset;
    tc.gait_left_ratio = cfg.gait_left_ratio;
    tc.gait_right_ratio = cfg.gait_right_ratio;
    tc.motion_length = cfg.motion_length;
    impl_->term_calc.Init(tc, cfg.rl_default_pos, cfg.action_joint_index, impl_->action_dim);

    // 初始化自定义标量
    for (const auto &kv : cfg.custom_scalar_defaults) {
        impl_->term_calc.SetCustomScalar(kv.first, kv.second);
    }

    // ---- 为每段创建 assembler ----
    for (const auto &seg_cfg : cfg.obs_segments) {
        Impl::SegmentRuntime seg;
        seg.seg_cfg = seg_cfg;

        // 计算每个 term 在帧内的布局
        int offset = 0;
        for (const auto &term_name : seg_cfg.terms) {
            int dim = impl_->term_calc.TermDim(term_name);
            seg.term_layouts.push_back({offset, dim});
            offset += dim;
        }
        seg.frame_dim = offset;

        // 创建 assembler 并初始化
        seg.assembler = CreateAssembler(seg_cfg.mode);
        seg.assembler->Init(seg.frame_dim,
                            seg.term_layouts,
                            seg_cfg.length,
                            seg_cfg.order,
                            seg_cfg.include_current);

        impl_->segments.push_back(std::move(seg));
    }

    // ---- 计算期望观测维度 ----
    impl_->expected_obs_dim = 0;
    for (const auto &seg : impl_->segments) {
        impl_->expected_obs_dim += seg.assembler->OutputDim();
    }

    // ---- 打印段信息 ----
    std::cout << "[PolicyExecutor] 段式观测: " << impl_->segments.size()
            << " 段, 计算维度=" << impl_->expected_obs_dim << ", 模型输入维度=" << impl_->obs_dim
            << std::endl;
    for (size_t i = 0; i < impl_->segments.size(); ++i) {
        const auto &s = impl_->segments[i];
        const auto &sc = s.seg_cfg;
        std::cout << "  段[" << i << "] mode=" << (sc.mode.empty() ? "(none)" : sc.mode)
                << ", frame_dim=" << s.frame_dim;
        if (!sc.mode.empty()) {
            std::cout << ", length=" << sc.length << ", order=" << sc.order
                    << ", include_current=" << sc.include_current;
        }
        std::cout << ", output_dim=" << s.assembler->OutputDim() << ", terms=[";
        for (size_t j = 0; j < sc.terms.size(); ++j) {
            if (j > 0)
                std::cout << ",";
            std::cout << sc.terms[j];
        }
        std::cout << "]" << std::endl;
    }

    // ---- 维度校验 ----
    if (cfg.strict_obs_dim_check && impl_->expected_obs_dim != impl_->obs_dim) {
        throw std::runtime_error(
            "[PolicyExecutor] 观测维度不匹配: 段计算=" + std::to_string(impl_->expected_obs_dim) +
            ", 模型输入=" + std::to_string(impl_->obs_dim));
    }
    if (!cfg.strict_obs_dim_check && impl_->expected_obs_dim != impl_->obs_dim) {
        std::cerr << "[PolicyExecutor] 警告: 观测维度不匹配（非严格模式）"
                << " 段计算=" << impl_->expected_obs_dim << ", 模型输入=" << impl_->obs_dim
                << std::endl;
    }

    impl_->initialized = true;
}

// ============================================================
// assembleObs -> AssembleObs
// ============================================================

void PolicyExecutor::AssembleObs(const std::array<double, 3> &gyro,
                                const std::array<double, 3> &rpy,
                                double cmd_vx,
                                double cmd_vy,
                                double cmd_wz,
                                const std::vector<double> &joint_pos,
                                const std::vector<double> &joint_vel,
                                const std::array<double, 4> &base_quat,
                                const std::array<double, 3> &base_vel,
                                float dt,
                                Eigen::VectorXf &out_obs) {
    if (!impl_->initialized || impl_->obs_dim <= 0)
        return;

    const std::array<double, 3> normalized_rpy = NormalizeRpy(rpy);

    if (out_obs.size() != impl_->obs_dim) {
        out_obs.setZero(impl_->obs_dim);
    } else {
        out_obs.setZero();
    }

    impl_->term_calc.AdvancePhase(dt);
    int out_idx = 0;

    for (auto &seg : impl_->segments) {
        // 1. 计算当前帧
        std::vector<float> frame(seg.frame_dim, 0.0f);
        for (size_t j = 0; j < seg.seg_cfg.terms.size(); ++j) {
            impl_->term_calc.FillTermValues(seg.seg_cfg.terms[j],
                                            gyro,
                                            normalized_rpy,
                                            cmd_vx,
                                            cmd_vy,
                                            cmd_wz,
                                            joint_pos,
                                            joint_vel,
                                            impl_->blended_action,
                                            base_quat,
                                            base_vel,
                                            frame.data() + seg.term_layouts[j].offset);
        }

        // 2. 交给 assembler
        const int seg_out = seg.assembler->OutputDim();
        if (out_idx + seg_out <= impl_->obs_dim) {
            seg.assembler->Assemble(frame.data(), out_obs.data() + out_idx);
        }
        out_idx += seg_out;
    }
}

// ============================================================
// Infer
// ============================================================

void PolicyExecutor::Infer(const Eigen::VectorXf &obs, std::vector<double> &out_action) {
    if (!impl_->initialized) {
        throw std::runtime_error("[PolicyExecutor] 未初始化");
    }

    if (obs.size() != impl_->obs_dim) {
        throw std::runtime_error(
            "[PolicyExecutor] 观测维度错误: 实际=" + std::to_string(obs.size()) +
            ", 期望=" + std::to_string(impl_->obs_dim));
    }

    impl_->onnx.GetInput(0) = obs;

    if (impl_->has_obs_hist) {
        if (static_cast<int>(impl_->obs_hist.size()) >= impl_->obs_hist_len) {
            impl_->obs_hist.erase(impl_->obs_hist.begin());
        }
        impl_->obs_hist.push_back(obs);

        auto &hist_input = impl_->onnx.GetInput(1);
        const int total_hist = impl_->obs_hist_len * impl_->obs_dim;
        if (hist_input.size() != total_hist)
            hist_input.resize(total_hist);
        for (int t = 0; t < impl_->obs_hist_len; ++t) {
            if (t < static_cast<int>(impl_->obs_hist.size())) {
                hist_input.segment(t * impl_->obs_dim, impl_->obs_dim) = impl_->obs_hist[t];
            } else {
                hist_input.segment(t * impl_->obs_dim, impl_->obs_dim) = obs;
            }
        }
    } else if (impl_->has_lstm) {
        impl_->onnx.GetInput(1) = impl_->h_state;
        impl_->onnx.GetInput(2) = impl_->c_state;
    }

    impl_->onnx.Run();

    if (impl_->has_lstm) {
        impl_->h_state = impl_->onnx.GetOutput(1);
        impl_->c_state = impl_->onnx.GetOutput(2);
    }

    const auto &out = impl_->onnx.GetOutput(0);
    out_action.resize(out.size());
    const double blend = impl_->cfg.action_blend_ratio;
    for (int i = 0; i < out.size(); ++i) {
        const double raw = static_cast<double>(out[i]);
        const double prev =
            (i < static_cast<int>(impl_->blended_action.size())) ? impl_->blended_action[i] : 0.0;
        const double blended = blend * raw + (1.0 - blend) * prev;
        out_action[i] = blended;
        if (i < static_cast<int>(impl_->blended_action.size())) {
            impl_->blended_action[i] = blended;
        }
    }
}

// ============================================================
// MapActionToTargetPos
// ============================================================

void PolicyExecutor::MapActionToTargetPos(const std::vector<double> &action,
                                        std::vector<double> &target_pos) const {
    const auto &c = impl_->cfg;
    const int ndof = static_cast<int>(c.rl_default_pos.size());
    target_pos.assign(ndof, 0.0);

    // 未被策略控制的关节保持 rl_default_pos
    for (int j = 0; j < ndof; ++j) {
        target_pos[j] = c.rl_default_pos[j];
    }

    // 策略动作覆盖映射关节
    const bool per_joint_scale = c.action_scale.size() > 1;
    if (!c.action_joint_index.empty()) {
        const int n_act =
            static_cast<int>(std::min<std::size_t>(action.size(), c.action_joint_index.size()));
        for (int i = 0; i < n_act; ++i) {
            const int j = c.action_joint_index[i];
            if (j < 0 || j >= ndof)
                continue;
            const double scale = per_joint_scale ? c.action_scale[i] : c.action_scale[0];
            target_pos[j] = action[i] * scale + c.rl_default_pos[j];
        }
    } else {
        const int n_act = std::min(ndof, static_cast<int>(action.size()));
        for (int i = 0; i < n_act; ++i) {
            const double scale = per_joint_scale ? c.action_scale[i] : c.action_scale[0];
            target_pos[i] = action[i] * scale + c.rl_default_pos[i];
        }
    }
}

}  // namespace rl_policy
