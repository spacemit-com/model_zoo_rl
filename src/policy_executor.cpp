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
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
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
    bool has_obs_hist = false;
    int obs_dim = 0;
    int action_dim = 0;
    int expected_obs_dim = 0;

    struct InputBindingRuntime {
        ModelInputBindingConfig binding;
        int input_index = -1;
        int observation_offset = -1;
        int feedback_output_index = -1;
        int history_source_runtime_index = -1;
        int history_length = 0;
        Eigen::VectorXf value;
        std::vector<Eigen::VectorXf> history;
        bool ready = false;
    };

    struct InputCompilation {
        std::set<std::string> used_inputs;
        std::set<std::string> feedback_outputs;
        std::unordered_map<std::string, size_t> runtime_by_name;
        std::vector<std::string> observation_inputs;
    };

    std::vector<InputBindingRuntime> input_bindings;
    int feedback_state_count = 0;
    int action_output_index = -1;
    std::unordered_map<std::string, size_t> external_input_bindings;
    std::unordered_map<std::string, int> exposed_output_indices;
    bool outputs_ready = false;

    // 内部维护的平滑动作，用于 last_action 与输出一致
    std::vector<double> blended_action;

    int FindInput(const std::string &name) const;
    int FindOutput(const std::string &name) const;
    void ResetModelIOState();
    InputCompilation CompileModelInputs();
    void CompileObservationHistories(const InputCompilation &compilation);
    void CompileModelOutputs(const std::set<std::string> &feedback_outputs);
    void ValidateActionConfig() const;
    void CompileModelIO();
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

void FillInitialValue(
    const std::vector<float> &values,
    Eigen::VectorXf &target,
    const std::string &name) {
    if (values.empty()) {
        throw std::runtime_error("[PolicyExecutor] 张量缺少初始值: " + name);
    }
    if (values.size() == 1) {
        target.setConstant(values.front());
        return;
    }
    if (values.size() != static_cast<size_t>(target.size())) {
        throw std::runtime_error(
            "[PolicyExecutor] 张量初始值维度错误: " + name + ", 实际=" +
            std::to_string(values.size()) + ", 期望=1 或 " + std::to_string(target.size()));
    }
    for (int i = 0; i < target.size(); ++i) target[i] = values[i];
}

void ValidateClip(const std::vector<double> &clip, int dim, const std::string &name) {
    if (clip.empty()) return;
    if (clip.size() != 1 && clip.size() != static_cast<size_t>(dim)) {
        throw std::runtime_error(
            "[PolicyExecutor] " + name + " 维度错误: 实际=" + std::to_string(clip.size()) +
            ", 期望=1 或 " + std::to_string(dim));
    }
    for (double value : clip) {
        if (!std::isfinite(value) || value <= 0.0) {
            throw std::runtime_error("[PolicyExecutor] " + name + " 必须是有限正数");
        }
    }
}

float ClipValue(float value, const std::vector<double> &clip, int index) {
    if (clip.empty()) return value;
    const double limit = clip.size() == 1 ? clip.front() : clip[index];
    return static_cast<float>(std::clamp(static_cast<double>(value), -limit, limit));
}

void ValidateBatchOne(
    const onnx_runtime::TensorInfo &info, const std::string &semantic) {
    if (info.shape.size() >= 2 && info.shape.front() != 1) {
        throw std::runtime_error(
            "[PolicyExecutor] " + semantic + " 仅支持 batch=1: " + info.name +
            " 的 batch=" + std::to_string(info.shape.front()));
    }
}

void ValidateFloatInput(
    const onnx_runtime::OnnxRuntimeClass &onnx,
    int input_index,
    const std::string &source) {
    if (!onnx.CanSetInputFromFloat(input_index)) {
        const auto &info = onnx.GetInputInfo(input_index);
        throw std::runtime_error(
            "[PolicyExecutor] " + source + " 输入不支持 float 语义转换: " +
            info.name + " (" + info.element_type_name + ")");
    }
}

onnx_runtime::TensorElementType ToBackendTensorType(TensorElementType type) {
    switch (type) {
    case TensorElementType::FLOAT32: return onnx_runtime::TensorElementType::FLOAT32;
    case TensorElementType::UINT8: return onnx_runtime::TensorElementType::UINT8;
    case TensorElementType::INT8: return onnx_runtime::TensorElementType::INT8;
    case TensorElementType::UINT16: return onnx_runtime::TensorElementType::UINT16;
    case TensorElementType::INT16: return onnx_runtime::TensorElementType::INT16;
    case TensorElementType::INT32: return onnx_runtime::TensorElementType::INT32;
    case TensorElementType::INT64: return onnx_runtime::TensorElementType::INT64;
    case TensorElementType::STRING: return onnx_runtime::TensorElementType::STRING;
    case TensorElementType::BOOL: return onnx_runtime::TensorElementType::BOOL;
    case TensorElementType::FLOAT16: return onnx_runtime::TensorElementType::FLOAT16;
    case TensorElementType::FLOAT64: return onnx_runtime::TensorElementType::FLOAT64;
    case TensorElementType::UINT32: return onnx_runtime::TensorElementType::UINT32;
    case TensorElementType::UINT64: return onnx_runtime::TensorElementType::UINT64;
    case TensorElementType::COMPLEX64: return onnx_runtime::TensorElementType::COMPLEX64;
    case TensorElementType::COMPLEX128: return onnx_runtime::TensorElementType::COMPLEX128;
    case TensorElementType::BFLOAT16: return onnx_runtime::TensorElementType::BFLOAT16;
    case TensorElementType::FLOAT8E4M3FN:
        return onnx_runtime::TensorElementType::FLOAT8E4M3FN;
    case TensorElementType::FLOAT8E4M3FNUZ:
        return onnx_runtime::TensorElementType::FLOAT8E4M3FNUZ;
    case TensorElementType::FLOAT8E5M2:
        return onnx_runtime::TensorElementType::FLOAT8E5M2;
    case TensorElementType::FLOAT8E5M2FNUZ:
        return onnx_runtime::TensorElementType::FLOAT8E5M2FNUZ;
    case TensorElementType::UINT4: return onnx_runtime::TensorElementType::UINT4;
    case TensorElementType::INT4: return onnx_runtime::TensorElementType::INT4;
    case TensorElementType::UNDEFINED:
    default:
        return onnx_runtime::TensorElementType::UNDEFINED;
    }
}

TensorElementType FromBackendTensorType(onnx_runtime::TensorElementType type) {
    switch (type) {
    case onnx_runtime::TensorElementType::FLOAT32: return TensorElementType::FLOAT32;
    case onnx_runtime::TensorElementType::UINT8: return TensorElementType::UINT8;
    case onnx_runtime::TensorElementType::INT8: return TensorElementType::INT8;
    case onnx_runtime::TensorElementType::UINT16: return TensorElementType::UINT16;
    case onnx_runtime::TensorElementType::INT16: return TensorElementType::INT16;
    case onnx_runtime::TensorElementType::INT32: return TensorElementType::INT32;
    case onnx_runtime::TensorElementType::INT64: return TensorElementType::INT64;
    case onnx_runtime::TensorElementType::STRING: return TensorElementType::STRING;
    case onnx_runtime::TensorElementType::BOOL: return TensorElementType::BOOL;
    case onnx_runtime::TensorElementType::FLOAT16: return TensorElementType::FLOAT16;
    case onnx_runtime::TensorElementType::FLOAT64: return TensorElementType::FLOAT64;
    case onnx_runtime::TensorElementType::UINT32: return TensorElementType::UINT32;
    case onnx_runtime::TensorElementType::UINT64: return TensorElementType::UINT64;
    case onnx_runtime::TensorElementType::COMPLEX64: return TensorElementType::COMPLEX64;
    case onnx_runtime::TensorElementType::COMPLEX128: return TensorElementType::COMPLEX128;
    case onnx_runtime::TensorElementType::BFLOAT16: return TensorElementType::BFLOAT16;
    case onnx_runtime::TensorElementType::FLOAT8E4M3FN:
        return TensorElementType::FLOAT8E4M3FN;
    case onnx_runtime::TensorElementType::FLOAT8E4M3FNUZ:
        return TensorElementType::FLOAT8E4M3FNUZ;
    case onnx_runtime::TensorElementType::FLOAT8E5M2:
        return TensorElementType::FLOAT8E5M2;
    case onnx_runtime::TensorElementType::FLOAT8E5M2FNUZ:
        return TensorElementType::FLOAT8E5M2FNUZ;
    case onnx_runtime::TensorElementType::UINT4: return TensorElementType::UINT4;
    case onnx_runtime::TensorElementType::INT4: return TensorElementType::INT4;
    case onnx_runtime::TensorElementType::UNDEFINED:
    default:
        return TensorElementType::UNDEFINED;
    }
}

}  // namespace

int PolicyExecutor::Impl::FindInput(const std::string &name) const {
    for (int i = 0; i < onnx.GetInputCount(); ++i) {
        if (onnx.GetInputInfo(i).name == name) return i;
    }
    return -1;
}

int PolicyExecutor::Impl::FindOutput(const std::string &name) const {
    for (int i = 0; i < onnx.GetOutputCount(); ++i) {
        if (onnx.GetOutputInfo(i).name == name) return i;
    }
    return -1;
}

void PolicyExecutor::Impl::ResetModelIOState() {
    obs_dim = 0;
    action_dim = 0;
    has_obs_hist = false;
    feedback_state_count = 0;
    action_output_index = -1;
    input_bindings.clear();
    external_input_bindings.clear();
    exposed_output_indices.clear();
    outputs_ready = false;
    blended_action.clear();
}

PolicyExecutor::Impl::InputCompilation PolicyExecutor::Impl::CompileModelInputs() {
    InputCompilation compilation;
    int next_observation_offset = 0;

    for (const auto &binding : cfg.model_io.inputs) {
        const int input_index = FindInput(binding.name);
        if (input_index < 0) {
            throw std::runtime_error("[PolicyExecutor] model_io 输入不存在: " + binding.name);
        }
        if (!compilation.used_inputs.insert(binding.name).second) {
            throw std::runtime_error(
                "[PolicyExecutor] model_io 输入重复绑定: " + binding.name);
        }

        const auto &input_info = onnx.GetInputInfo(input_index);
        InputBindingRuntime runtime;
        runtime.binding = binding;
        runtime.input_index = input_index;
        runtime.value.setZero(static_cast<int>(input_info.total_size));

        switch (binding.source) {
        case ModelInputSource::OBSERVATION: {
            ValidateBatchOne(input_info, "observation");
            ValidateFloatInput(onnx, input_index, "observation");
            const int offset = binding.observation_offset >= 0
                ? binding.observation_offset : next_observation_offset;
            runtime.observation_offset = offset;
            next_observation_offset = std::max(
                next_observation_offset,
                offset + static_cast<int>(input_info.total_size));
            obs_dim = std::max(
                obs_dim, offset + static_cast<int>(input_info.total_size));
            compilation.observation_inputs.push_back(binding.name);
            runtime.ready = true;
            break;
        }
        case ModelInputSource::FEEDBACK: {
            if (binding.feedback_output.empty()) {
                throw std::runtime_error(
                    "[PolicyExecutor] feedback 输入缺少 output: " + binding.name);
            }
            const int output_index = FindOutput(binding.feedback_output);
            if (output_index < 0) {
                throw std::runtime_error(
                    "[PolicyExecutor] feedback 输出不存在: " + binding.feedback_output);
            }
            const auto &output_info = onnx.GetOutputInfo(output_index);
            if (input_info.total_size != output_info.total_size) {
                throw std::runtime_error(
                    "[PolicyExecutor] feedback 输入输出元素数不一致: " + binding.name +
                    " <- " + binding.feedback_output);
            }
            if (input_info.element_type != output_info.element_type) {
                throw std::runtime_error(
                    "[PolicyExecutor] feedback 输入输出 dtype 不一致: " + binding.name +
                    " <- " + binding.feedback_output);
            }
            if (!binding.initial_value.empty()) {
                FillInitialValue(binding.initial_value, runtime.value, binding.name);
            }
            compilation.feedback_outputs.insert(binding.feedback_output);
            runtime.feedback_output_index = output_index;
            runtime.ready = true;
            ++feedback_state_count;
            break;
        }
        case ModelInputSource::CONSTANT:
            FillInitialValue(binding.initial_value, runtime.value, binding.name);
            runtime.ready = true;
            break;
        case ModelInputSource::EXTERNAL: {
            if (!binding.initial_value.empty()) {
                FillInitialValue(binding.initial_value, runtime.value, binding.name);
                runtime.ready = true;
            }
            const std::string key = binding.key.empty() ? binding.name : binding.key;
            if (external_input_bindings.count(key) > 0) {
                throw std::runtime_error("[PolicyExecutor] external 输入 key 重复: " + key);
            }
            external_input_bindings[key] = input_bindings.size();
            break;
        }
        case ModelInputSource::OBSERVATION_HISTORY:
            ValidateBatchOne(input_info, "observation_history");
            ValidateFloatInput(onnx, input_index, "observation_history");
            runtime.ready = true;
            break;
        }

        // ONNX backend 已按原生 dtype 将输入清零。空初值的 feedback 直接复用该
        // 缓冲区，避免 FP8/INT4 等无 float 自动转换路径的状态张量初始化失败。
        if (!binding.initial_value.empty() &&
            (binding.source == ModelInputSource::FEEDBACK ||
            binding.source == ModelInputSource::CONSTANT ||
            binding.source == ModelInputSource::EXTERNAL)) {
            onnx.SetInputFromFloat(
                runtime.input_index, runtime.value.data(), runtime.value.size());
        }

        compilation.runtime_by_name[binding.name] = input_bindings.size();
        input_bindings.push_back(std::move(runtime));
    }

    for (int i = 0; i < onnx.GetInputCount(); ++i) {
        const auto &name = onnx.GetInputInfo(i).name;
        if (compilation.used_inputs.count(name) == 0) {
            throw std::runtime_error("[PolicyExecutor] 未绑定的 ONNX 输入: " + name);
        }
    }
    if (compilation.observation_inputs.empty()) {
        throw std::runtime_error("[PolicyExecutor] model_io 至少需要一个 observation 输入");
    }
    return compilation;
}

void PolicyExecutor::Impl::CompileObservationHistories(
    const InputCompilation &compilation) {
    for (auto &runtime : input_bindings) {
        if (runtime.binding.source != ModelInputSource::OBSERVATION_HISTORY) continue;

        std::string source = runtime.binding.history_source;
        if (source.empty() && compilation.observation_inputs.size() == 1) {
            source = compilation.observation_inputs.front();
        }
        if (source.empty()) {
            throw std::runtime_error(
                "[PolicyExecutor] observation_history 缺少 history_of: " +
                runtime.binding.name);
        }
        const auto source_it = compilation.runtime_by_name.find(source);
        if (source_it == compilation.runtime_by_name.end()) {
            throw std::runtime_error(
                "[PolicyExecutor] history_of 输入不存在: " + runtime.binding.name +
                " <- " + source);
        }
        auto &source_runtime = input_bindings[source_it->second];
        if (source_runtime.binding.source != ModelInputSource::OBSERVATION) {
            throw std::runtime_error(
                "[PolicyExecutor] observation_history 只能跟踪 observation 输入: " +
                runtime.binding.name + " <- " + source);
        }
        const int frame_size = static_cast<int>(source_runtime.value.size());
        if (frame_size <= 0 || runtime.value.size() <= 0 ||
            runtime.value.size() % frame_size != 0) {
            throw std::runtime_error(
                "[PolicyExecutor] observation_history 形状无法匹配源输入: " +
                runtime.binding.name + " <- " + source);
        }
        runtime.history_source_runtime_index = static_cast<int>(source_it->second);
        runtime.history_length = static_cast<int>(runtime.value.size()) / frame_size;
        runtime.history.assign(runtime.history_length, Eigen::VectorXf::Zero(frame_size));
        has_obs_hist = true;
    }
}

void PolicyExecutor::Impl::CompileModelOutputs(
    const std::set<std::string> &feedback_outputs) {
    std::set<std::string> declared_outputs;

    for (const auto &binding : cfg.model_io.outputs) {
        const int output_index = FindOutput(binding.name);
        if (output_index < 0) {
            throw std::runtime_error("[PolicyExecutor] model_io 输出不存在: " + binding.name);
        }
        if (!declared_outputs.insert(binding.name).second) {
            throw std::runtime_error(
                "[PolicyExecutor] model_io 输出重复声明: " + binding.name);
        }
        const auto &output_info = onnx.GetOutputInfo(output_index);
        switch (binding.target) {
        case ModelOutputTarget::ACTION:
            if (action_output_index >= 0) {
                throw std::runtime_error("[PolicyExecutor] model_io 只能配置一个 action 输出");
            }
            ValidateBatchOne(output_info, "action");
            if (!onnx.CanGetOutputAsFloat(output_index)) {
                throw std::runtime_error(
                    "[PolicyExecutor] action 输出不支持 float 语义转换: " +
                    output_info.name + " (" + output_info.element_type_name + ")");
            }
            action_output_index = output_index;
            action_dim = static_cast<int>(output_info.total_size);
            break;
        case ModelOutputTarget::EXPOSE: {
            const std::string key = binding.key.empty() ? binding.name : binding.key;
            if (exposed_output_indices.count(key) > 0) {
                throw std::runtime_error("[PolicyExecutor] expose 输出 key 重复: " + key);
            }
            exposed_output_indices[key] = output_index;
            break;
        }
        case ModelOutputTarget::IGNORE:
            break;
        }
    }

    if (action_output_index < 0) {
        throw std::runtime_error("[PolicyExecutor] model_io 缺少 action 输出");
    }
    for (int i = 0; i < onnx.GetOutputCount(); ++i) {
        const auto &name = onnx.GetOutputInfo(i).name;
        if (feedback_outputs.count(name) == 0 && declared_outputs.count(name) == 0) {
            throw std::runtime_error("[PolicyExecutor] 未绑定的 ONNX 输出: " + name);
        }
    }
}

void PolicyExecutor::Impl::ValidateActionConfig() const {
    if (cfg.action_scale.empty()) {
        throw std::runtime_error("[PolicyExecutor] action_scale 不能为空");
    }
    if (cfg.action_scale.size() != 1 &&
        cfg.action_scale.size() != static_cast<size_t>(action_dim)) {
        throw std::runtime_error(
            "[PolicyExecutor] action_scale 维度错误: 实际=" +
            std::to_string(cfg.action_scale.size()) + ", 期望=1 或 " +
            std::to_string(action_dim));
    }
    if (!cfg.action_joint_index.empty() &&
        cfg.action_joint_index.size() != static_cast<size_t>(action_dim)) {
        throw std::runtime_error(
            "[PolicyExecutor] action_joint_index 维度错误: 实际=" +
            std::to_string(cfg.action_joint_index.size()) + ", 期望=" +
            std::to_string(action_dim));
    }
    if (cfg.action_joint_index.empty() &&
        action_dim > static_cast<int>(cfg.rl_default_pos.size())) {
        throw std::runtime_error(
            "[PolicyExecutor] action 维度大于关节数，必须配置 action_joint_index");
    }

    std::set<int> mapped_joints;
    for (int index : cfg.action_joint_index) {
        if (index < 0 || index >= static_cast<int>(cfg.rl_default_pos.size())) {
            throw std::runtime_error(
                "[PolicyExecutor] action_joint_index 越界: " + std::to_string(index));
        }
        if (!mapped_joints.insert(index).second) {
            throw std::runtime_error(
                "[PolicyExecutor] action_joint_index 重复: " + std::to_string(index));
        }
    }
}

void PolicyExecutor::Impl::CompileModelIO() {
    if (cfg.model_io.inputs.empty() || cfg.model_io.outputs.empty()) {
        throw std::runtime_error(
            "[PolicyExecutor] model_io 必须包含非空 inputs 和 outputs");
    }
    if (onnx.GetInputCount() == 0 || onnx.GetOutputCount() == 0) {
        throw std::runtime_error("[PolicyExecutor] ONNX 模型至少需要一个输入和一个输出");
    }

    const InputCompilation compilation = CompileModelInputs();
    CompileObservationHistories(compilation);
    CompileModelOutputs(compilation.feedback_outputs);
    ValidateActionConfig();
    ValidateClip(cfg.clip_observations, obs_dim, "clip_observations");
    ValidateClip(cfg.clip_actions, action_dim, "clip_actions");

    std::cout << "[PolicyExecutor] model_io: 严格声明"
        << ", obs_dim=" << obs_dim << ", action_dim=" << action_dim
        << ", feedback=" << feedback_state_count
        << ", history=" << (has_obs_hist ? "yes" : "no")
        << ", external=" << external_input_bindings.size()
        << ", expose=" << exposed_output_indices.size() << std::endl;
}

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
int PolicyExecutor::FeedbackStateCount() const {
    return impl_->feedback_state_count;
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

// ---- 自定义数组（泛型 N 维 obs term 注入通道）----

void PolicyExecutor::SetCustomArray(const std::string &name, const float *data, int size) {
    impl_->term_calc.SetCustomArray(name, data, size);
}

void PolicyExecutor::SetModelInput(const std::string &key, const float *data, int size) {
    if (!impl_->initialized) {
        throw std::runtime_error("[PolicyExecutor] 未初始化");
    }
    const auto it = impl_->external_input_bindings.find(key);
    if (it == impl_->external_input_bindings.end()) {
        throw std::runtime_error("[PolicyExecutor] external 输入不存在: " + key);
    }
    auto &runtime = impl_->input_bindings[it->second];
    if (!data || size != runtime.value.size()) {
        throw std::runtime_error(
            "[PolicyExecutor] external 输入维度错误: " + key + ", 实际=" +
            std::to_string(size) + ", 期望=" + std::to_string(runtime.value.size()));
    }
    runtime.value = Eigen::Map<const Eigen::VectorXf>(data, size);
    impl_->onnx.SetInputFromFloat(runtime.input_index, data, size);
    runtime.ready = true;
}

void PolicyExecutor::SetModelInput(const std::string &key, const TensorView &input) {
    if (!impl_->initialized) {
        throw std::runtime_error("[PolicyExecutor] 未初始化");
    }
    const auto it = impl_->external_input_bindings.find(key);
    if (it == impl_->external_input_bindings.end()) {
        throw std::runtime_error("[PolicyExecutor] external 输入不存在: " + key);
    }
    auto &runtime = impl_->input_bindings[it->second];
    if (input.element_count != static_cast<size_t>(runtime.value.size())) {
        throw std::runtime_error(
            "[PolicyExecutor] external 输入维度错误: " + key + ", 实际=" +
            std::to_string(input.element_count) + ", 期望=" +
            std::to_string(runtime.value.size()));
    }
    impl_->onnx.SetInput(
        runtime.input_index,
        {
            ToBackendTensorType(input.element_type),
            input.data,
            input.element_count,
            input.byte_count,
        });
    runtime.ready = true;
}

const std::vector<float> &PolicyExecutor::GetModelOutput(const std::string &key) const {
    const auto it = impl_->exposed_output_indices.find(key);
    if (it == impl_->exposed_output_indices.end()) {
        throw std::runtime_error("[PolicyExecutor] expose 输出不存在: " + key);
    }
    if (!impl_->outputs_ready) {
        throw std::runtime_error("[PolicyExecutor] 尚未完成推理，输出不可用: " + key);
    }
    return impl_->onnx.GetOutput(it->second);
}

TensorView PolicyExecutor::GetModelOutputTensor(const std::string &key) const {
    const auto it = impl_->exposed_output_indices.find(key);
    if (it == impl_->exposed_output_indices.end()) {
        throw std::runtime_error("[PolicyExecutor] expose 输出不存在: " + key);
    }
    if (!impl_->outputs_ready) {
        throw std::runtime_error("[PolicyExecutor] 尚未完成推理，输出不可用: " + key);
    }
    const auto output = impl_->onnx.GetOutputView(it->second);
    return {
        FromBackendTensorType(output.element_type),
        output.data,
        output.element_count,
        output.byte_count,
    };
}

// ============================================================
// init
// ============================================================

void PolicyExecutor::Init(const PolicyExecutorConfig &cfg) {
    impl_->initialized = false;
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

    impl_->ResetModelIOState();
    impl_->CompileModelIO();

    // 初始化内部动作缓冲
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

    // 注册自定义数组维度（泛型 N 维 obs term，必须在 segment 构建前注册以便 TermDim 正确返回）
    for (const auto &kv : cfg.custom_array_dims) {
        impl_->term_calc.RegisterCustomArrayDim(kv.first, kv.second);
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
    impl_->outputs_ready = false;

    Eigen::VectorXf clipped_obs = obs;
    for (int i = 0; i < clipped_obs.size(); ++i) {
        clipped_obs[i] = ClipValue(clipped_obs[i], impl_->cfg.clip_observations, i);
    }

    for (auto &runtime : impl_->input_bindings) {
        switch (runtime.binding.source) {
        case ModelInputSource::OBSERVATION: {
            runtime.value = clipped_obs.segment(
                runtime.observation_offset, runtime.value.size());
            impl_->onnx.SetInputFromFloat(
                runtime.input_index, runtime.value.data(), runtime.value.size());
            break;
        }
        case ModelInputSource::FEEDBACK:
        case ModelInputSource::CONSTANT:
            break;
        case ModelInputSource::EXTERNAL:
            if (!runtime.ready) {
                const std::string key = runtime.binding.key.empty()
                    ? runtime.binding.name : runtime.binding.key;
                throw std::runtime_error(
                    "[PolicyExecutor] external 输入尚未设置且无默认值: " + key);
            }
            break;
        case ModelInputSource::OBSERVATION_HISTORY:
            break;
        }
    }

    for (auto &runtime : impl_->input_bindings) {
        if (runtime.binding.source != ModelInputSource::OBSERVATION_HISTORY) continue;
        const auto &source =
            impl_->input_bindings[runtime.history_source_runtime_index].value;
        runtime.history.erase(runtime.history.begin());
        runtime.history.push_back(source);
        for (int t = 0; t < runtime.history_length; ++t) {
            runtime.value.segment(t * source.size(), source.size()) = runtime.history[t];
        }
        impl_->onnx.SetInputFromFloat(
            runtime.input_index, runtime.value.data(), runtime.value.size());
    }

    if (!impl_->onnx.Run()) {
        throw std::runtime_error(
            "[PolicyExecutor] ONNX 推理失败: " + impl_->onnx.GetLastError());
    }

    for (auto &runtime : impl_->input_bindings) {
        if (runtime.binding.source == ModelInputSource::FEEDBACK) {
            impl_->onnx.CopyOutputToInput(
                runtime.feedback_output_index, runtime.input_index);
        }
    }
    impl_->outputs_ready = true;

    const auto &out = impl_->onnx.GetOutput(impl_->action_output_index);
    out_action.resize(out.size());
    const double blend = impl_->cfg.action_blend_ratio;
    for (int i = 0; i < out.size(); ++i) {
        const double raw = static_cast<double>(ClipValue(out[i], impl_->cfg.clip_actions, i));
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
