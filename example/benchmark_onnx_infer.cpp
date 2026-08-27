/**
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file benchmark_onnx_infer.cpp
 * @brief 使用 synthetic 输入测量 ONNX backend step 与周期调度
 *
 * 该工具按 model_io 生成或回灌 synthetic tensor，只测量当前 ONNX backend。
 * feedback 会逐轮回灌，constant 使用 YAML 配置；external 没有默认值时使用
 * synthetic 数据并在输出中明确标记。
 */

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "benchmark_common.h"
#include "onnx_infer.h"
#include "rl_service.h"

namespace {

using onnx_runtime::OnnxRuntimeClass;
using onnx_runtime::RuntimeOptions;
using onnx_runtime::TensorElementType;
using rl_benchmark::Clock;
using rl_benchmark::Options;
using rl_benchmark::Stats;
using rl_benchmark::TimingResult;
using rl_policy::LoadedPolicyConfig;
using rl_policy::ModelInputBindingConfig;
using rl_policy::ModelInputSource;
using rl_policy::ModelOutputTarget;

struct PreparedInput {
    ModelInputBindingConfig binding;
    int input_index = -1;
    int feedback_output_index = -1;
    bool update_synthetic = false;
    std::vector<float> values;
};

bool IsIntegerType(TensorElementType type) {
    switch (type) {
    case TensorElementType::UINT8:
    case TensorElementType::INT8:
    case TensorElementType::UINT16:
    case TensorElementType::INT16:
    case TensorElementType::INT32:
    case TensorElementType::INT64:
    case TensorElementType::BOOL:
    case TensorElementType::UINT32:
    case TensorElementType::UINT64:
        return true;
    default:
        return false;
    }
}

void FillConfigured(
    const std::vector<float> &configured, std::vector<float> *target, const std::string &name) {
    if (configured.empty()) {
        throw std::runtime_error("input requires a configured value: " + name);
    }
    if (configured.size() == 1) {
        std::fill(target->begin(), target->end(), configured.front());
        return;
    }
    if (configured.size() != target->size()) {
        throw std::runtime_error("input value size mismatch for " + name
            + ": actual=" + std::to_string(configured.size()) + ", expected=1 or "
            + std::to_string(target->size()));
    }
    *target = configured;
}

void FillSynthetic(
    const onnx_runtime::TensorInfo &info, std::uint64_t step, std::vector<float> *target) {
    const bool integer = IsIntegerType(info.element_type);
    if (info.element_type == TensorElementType::INT64 && target->size() == 1) {
        // 单元素 INT64 常用于序列索引。按逻辑 release 单调推进，使 drop
        // 之后的 synthetic 输入仍与调度时间线一致。
        (*target)[0] = static_cast<float>(step);
        return;
    }
    for (std::size_t i = 0; i < target->size(); ++i) {
        if (integer) {
            (*target)[i] = static_cast<float>((step + i) % 7);
        } else {
            const int centered = static_cast<int>(i % 23) - 11;
            (*target)[i]
                = static_cast<float>(centered) * 0.01f + static_cast<float>(step % 1000) * 0.00001f;
        }
    }
}

class BackendInputState {
public:
    BackendInputState(OnnxRuntimeClass *ort, const std::vector<ModelInputBindingConfig> &bindings)
        : ort_(ort) {
        std::unordered_map<std::string, ModelInputBindingConfig> by_name;
        for (const auto &binding : bindings) {
            if (!by_name.emplace(binding.name, binding).second) {
                throw std::runtime_error("duplicate model_io input: " + binding.name);
            }
        }

        for (int input_index = 0; input_index < ort_->GetInputCount(); ++input_index) {
            const auto &info = ort_->GetInputInfo(input_index);
            const auto binding_it = by_name.find(info.name);
            if (binding_it == by_name.end()) {
                throw std::runtime_error("ONNX input is missing from YAML model_io: " + info.name);
            }

            PreparedInput input;
            input.binding = binding_it->second;
            input.input_index = input_index;
            input.values.resize(static_cast<std::size_t>(info.total_size), 0.0f);
            PrepareInitialValue(info, &input);
            prepared_.push_back(std::move(input));
        }

        if (prepared_.size() != bindings.size()) {
            throw std::runtime_error("YAML model_io contains inputs not present in the ONNX model");
        }
    }

    void PrepareStep(std::uint64_t step) {
        for (auto &input : prepared_) {
            if (!input.update_synthetic)
                continue;
            const auto &info = ort_->GetInputInfo(input.input_index);
            FillSynthetic(info, step, &input.values);
            ort_->SetInputFromFloat(input.input_index, input.values.data(), input.values.size());
        }
    }

    void ApplyFeedback() {
        for (const auto &input : prepared_) {
            if (input.feedback_output_index < 0)
                continue;
            ort_->CopyOutputToInput(input.feedback_output_index, input.input_index);
        }
    }

    int SyntheticExternalCount() const { return synthetic_external_count_; }
    int DefaultExternalCount() const { return default_external_count_; }
    int NativeZeroInputCount() const { return native_zero_input_count_; }
    int FeedbackCount() const { return feedback_count_; }
    const std::vector<std::string> &SyntheticExternalNames() const {
        return synthetic_external_names_;
    }

private:
    int FindOutput(const std::string &name) const {
        for (int i = 0; i < ort_->GetOutputCount(); ++i) {
            if (ort_->GetOutputInfo(i).name == name)
                return i;
        }
        return -1;
    }

    void PrepareInitialValue(const onnx_runtime::TensorInfo &info, PreparedInput *input) {
        bool should_write = true;
        switch (input->binding.source) {
        case ModelInputSource::OBSERVATION:
        case ModelInputSource::OBSERVATION_HISTORY:
            FillSynthetic(info, 0, &input->values);
            input->update_synthetic = true;
            break;
        case ModelInputSource::FEEDBACK:
            ++feedback_count_;
            if (!input->binding.initial_value.empty()) {
                FillConfigured(input->binding.initial_value, &input->values, info.name);
            }
            input->feedback_output_index = FindOutput(input->binding.feedback_output);
            if (input->feedback_output_index < 0) {
                throw std::runtime_error(
                    "feedback output is missing: " + input->binding.feedback_output);
            }
            break;
        case ModelInputSource::CONSTANT:
            FillConfigured(input->binding.initial_value, &input->values, info.name);
            break;
        case ModelInputSource::EXTERNAL:
            if (input->binding.initial_value.empty()) {
                FillSynthetic(info, 0, &input->values);
                input->update_synthetic = true;
                ++synthetic_external_count_;
                synthetic_external_names_.push_back(
                    input->binding.key.empty() ? input->binding.name : input->binding.key);
            } else {
                FillConfigured(input->binding.initial_value, &input->values, info.name);
                ++default_external_count_;
            }
            break;
        }

        if (ort_->CanSetInputFromFloat(input->input_index)) {
            ort_->SetInputFromFloat(input->input_index, input->values.data(), input->values.size());
        } else {
            should_write = false;
            input->update_synthetic = false;
            ++native_zero_input_count_;
        }

        if (!should_write && !input->binding.initial_value.empty()) {
            throw std::runtime_error("cannot apply configured float value to native-only input: "
                + info.name + " (dtype=" + info.element_type_name + ")");
        }
    }

    OnnxRuntimeClass *ort_;
    std::vector<PreparedInput> prepared_;
    int synthetic_external_count_ = 0;
    int default_external_count_ = 0;
    int native_zero_input_count_ = 0;
    int feedback_count_ = 0;
    std::vector<std::string> synthetic_external_names_;
};

void PrintUsage(const char *program) {
    std::cerr << "用法: " << program << " <yaml_path> <policy_name> <robot_dir> [options]\n";
    rl_benchmark::PrintCommonUsage(std::cerr);
}

void PrintStats(const std::string &label, const Stats &stats) {
    std::cout << std::left << std::setw(22) << label << " min=" << std::right << std::fixed
            << std::setprecision(3) << stats.min << " avg=" << stats.avg
            << " std=" << stats.std_dev << " p50=" << stats.p50 << " p95=" << stats.p95
            << " p99=" << stats.p99;
    if (stats.has_p999)
        std::cout << " p99.9=" << stats.p999;
    if (stats.has_p9999)
        std::cout << " p99.99=" << stats.p9999;
    std::cout << " max=" << stats.max << " ms\n";
}

void WriteCsv(const std::string &path,
    const std::string &policy_name,
    const std::string &model_path,
    const Options &options,
    const std::string &affinity_before_init,
    const std::string &affinity_after_init,
    const onnx_runtime::RuntimeInfo &runtime,
    const TimingResult &timing,
    const std::vector<double> &backend_run_ms,
    const std::vector<float> &final_action) {
    if (path.empty())
        return;
    auto output = rl_benchmark::OpenCsv(path);
    output << "# benchmark=onnx_backend_synthetic\n"
        << "# policy=" << policy_name << "\n"
        << "# model=" << model_path << "\n"
        << "# build_type=" << RL_BENCHMARK_BUILD_TYPE << "\n"
        << "# cxx_flags=" << RL_BENCHMARK_CXX_FLAGS << "\n"
        << "# mode=" << rl_benchmark::ModeName(options.mode) << "\n"
        << "# hz=" << options.hz << "\n"
        << "# provider_requested=" << runtime.requested_provider << "\n"
        << "# provider_initialized=" << runtime.initialized_provider << "\n"
        << "# provider_status=" << runtime.provider_status << "\n"
        << "# threads=" << options.threads << "\n"
        << "# ep_threads_requested=" << runtime.ep_threads << "\n"
        << "# ort_spinning=" << (runtime.ort_spinning ? "on" : "off") << "\n"
        << "# affinity_requested=" << options.affinity << "\n"
        << "# affinity_effective_before_init=" << affinity_before_init << "\n"
        << "# affinity_effective_after_init=" << affinity_after_init << "\n"
        << "# affinity_ep_requested=" << runtime.affinity << "\n"
        << "# ep_dump_subgraphs=" << (options.ep_dump_subgraphs ? "true" : "false") << "\n"
        << "# ep_profile_prefix=" << options.ep_profile_prefix << "\n";
    output << "# measure_start_delay_ms=" << options.measure_start_delay_ms << "\n";
    rl_benchmark::WriteVectorEvidence(output, "final_action", final_action);
    output << "iteration,release_index,dropped_before,backlog_releases,"
            "backend_run_ms,backend_step_ms,release_jitter_ms,response_ms,"
            "deadline_lateness_ms\n";
    for (std::size_t i = 0; i < timing.samples.size(); ++i) {
        const auto &sample = timing.samples[i];
        output << i << ',' << sample.release_index << ',' << sample.dropped_before << ','
            << sample.backlog_releases << ',' << backend_run_ms[i] << ',' << sample.service_ms
            << ',' << sample.release_jitter_ms << ',' << sample.response_ms << ','
            << sample.deadline_lateness_ms << '\n';
    }
}

std::vector<float> ReadActionOutput(
    OnnxRuntimeClass *ort, const LoadedPolicyConfig &loaded) {
    for (const auto &binding : loaded.exec_cfg.model_io.outputs) {
        if (binding.target != ModelOutputTarget::ACTION)
            continue;
        for (int index = 0; index < ort->GetOutputCount(); ++index) {
            if (ort->GetOutputInfo(index).name == binding.name
                && ort->CanGetOutputAsFloat(index)) {
                return ort->GetOutput(index);
            }
        }
    }
    return {};
}

void PrintVerbose(const TimingResult &timing, const std::vector<double> &backend_run_ms) {
    for (std::size_t i = 0; i < timing.samples.size(); ++i) {
        const auto &sample = timing.samples[i];
        std::cout << "sample=" << i << " release=" << sample.release_index
                << " run_ms=" << backend_run_ms[i] << " step_ms=" << sample.service_ms;
        if (sample.response_ms != sample.service_ms) {
            std::cout << " jitter_ms=" << sample.release_jitter_ms
                    << " response_ms=" << sample.response_ms
                    << " lateness_ms=" << sample.deadline_lateness_ms;
        }
        std::cout << '\n';
    }
}

}  // namespace

int main(int argc, char *argv[]) {
    if (argc < 4) {
        PrintUsage(argv[0]);
        return 1;
    }

    try {
        const std::string yaml_path = argv[1];
        const std::string policy_name = argv[2];
        const std::string robot_dir = argv[3];
        const LoadedPolicyConfig loaded
            = rl_policy::LoadPolicyConfigFromYaml(yaml_path, policy_name, robot_dir);
        const Options options = rl_benchmark::ParseOptions(argc, argv, 4, 1.0 / loaded.rl_dt);
        const double input_dt = options.hz_overridden ? 1.0 / options.hz : loaded.rl_dt;
        const std::string affinity_before_init
            = rl_benchmark::ApplyAndGetAffinity(options.affinity);
        const std::string requested_ep_affinity = options.ep_affinity.empty()
            ? options.affinity
            : options.ep_affinity;
        const std::string ep_affinity = rl_benchmark::EpAffinityFromCpuList(
            requested_ep_affinity,
            options.threads,
            rl_benchmark::ShouldConfigureSpaceMITProvider(options.provider));

        OnnxRuntimeClass ort;
        RuntimeOptions runtime_options;
        runtime_options.provider = options.provider;
        runtime_options.threads = options.threads;
        runtime_options.affinity = ep_affinity;
        runtime_options.ep_dump_subgraphs = options.ep_dump_subgraphs;
        runtime_options.ep_profile_prefix = options.ep_profile_prefix;
        runtime_options.ort_spinning = options.ort_spinning;
        if (!ort.Init(loaded.exec_cfg.model_path, runtime_options)) {
            throw std::runtime_error("model init failed: " + ort.GetLastError());
        }
        const std::string affinity_after_init = rl_benchmark::ApplyAndGetAffinity("");
        BackendInputState input_state(&ort, loaded.exec_cfg.model_io.inputs);
        const auto &runtime = ort.GetRuntimeInfo();

        std::cout << "\nONNX backend synthetic benchmark\n"
                << "Policy:      " << policy_name << "\n"
                << "Model:       " << loaded.exec_cfg.model_path << "\n"
                << "Semantics:   synthetic backend step; not robot pipeline\n"
                << "Mode:        " << rl_benchmark::ModeName(options.mode) << "\n"
                << "Frequency:   " << options.hz << " Hz (input dt=" << input_dt << " s)\n"
                << "Provider:    requested=" << runtime.requested_provider
                << ", initialized=" << runtime.initialized_provider << "\n"
                << "EP status:   " << runtime.provider_status << "\n"
                << "Threads:     ORT intra=" << runtime.ort_intra_threads
                << ", ORT inter=" << runtime.ort_inter_threads
                << ", EP requested=" << runtime.ep_threads
                << "\n"
                << "ORT spin:    " << (runtime.ort_spinning ? "on" : "off")
                << " (intra-op only; does not control SpaceMIT EP workers)\n"
                << "Host affinity: requested="
                << (options.affinity.empty() ? "not-set" : options.affinity)
                << ", before-init=" << affinity_before_init
                << ", after-init=" << affinity_after_init << "\n"
                << "EP affinity requested: "
                << (runtime.affinity.empty() ? "not-set" : runtime.affinity)
                << "\n"
                << "EP diagnose: dump=" << (options.ep_dump_subgraphs ? "on" : "off")
                << ", profile="
                << (options.ep_profile_prefix.empty() ? "off" : options.ep_profile_prefix) << "\n"
                << "Feedback:    " << input_state.FeedbackCount() << "\n"
                << "External:    synthetic=" << input_state.SyntheticExternalCount()
                << ", configured-default=" << input_state.DefaultExternalCount() << "\n"
                << "Native zero: " << input_state.NativeZeroInputCount() << "\n"
                << "Warmup/test: " << options.warmup << '/' << options.rounds << "\n";
        rl_benchmark::PrintBuildMetadata();
        std::cout << "Thread count and affinity are configuration evidence, not proof that "
                    "every allowed CPU stayed busy.\n";
        if (!input_state.SyntheticExternalNames().empty()) {
            std::cout << "Synthetic external keys:";
            for (const auto &name : input_state.SyntheticExternalNames()) {
                std::cout << ' ' << name;
            }
            std::cout << "\n";
        }

        const auto run_step =
            [&](std::uint64_t synthetic_step, bool measured, std::vector<double> *run_ms) {
            input_state.PrepareStep(synthetic_step);
            const auto run_start = Clock::now();
            if (!ort.Run()) {
                throw std::runtime_error("ONNX Run failed: " + ort.GetLastError());
            }
            const auto run_finish = Clock::now();
            input_state.ApplyFeedback();
            if (measured) {
                run_ms->push_back(rl_benchmark::ToMilliseconds(run_finish - run_start));
            }
        };

        std::cout << "Warmup...\n" << std::flush;
        std::vector<double> backend_run_ms;
        backend_run_ms.reserve(options.rounds);
        for (int i = 0; i < options.warmup; ++i) {
            run_step(static_cast<std::uint64_t>(i), false, &backend_run_ms);
        }

        rl_benchmark::BeginMeasuredRegion(options);
        const TimingResult timing = rl_benchmark::MeasureRounds(
            options, [&](std::uint64_t release_index) {
                run_step(static_cast<std::uint64_t>(options.warmup) + release_index,
                    true,
                    &backend_run_ms);
            });
        rl_benchmark::EndMeasuredRegion();

        PrintStats("ONNX Run", rl_benchmark::ComputeStats(backend_run_ms));
        PrintStats("Backend step",
            rl_benchmark::ComputeStats(rl_benchmark::ExtractMetric(
                timing.samples, &rl_benchmark::TimingSample::service_ms)));

        if (options.mode == rl_benchmark::Mode::PERIODIC) {
            PrintStats("Release jitter",
                rl_benchmark::ComputeStats(rl_benchmark::ExtractMetric(
                    timing.samples, &rl_benchmark::TimingSample::release_jitter_ms)));
            PrintStats("Release-to-done",
                rl_benchmark::ComputeStats(rl_benchmark::ExtractMetric(
                    timing.samples, &rl_benchmark::TimingSample::response_ms)));
            PrintStats("Deadline lateness",
                rl_benchmark::ComputeStats(rl_benchmark::ExtractMetric(
                    timing.samples, &rl_benchmark::TimingSample::deadline_lateness_ms)));
            std::cout << "Periodic deadline: misses=" << timing.deadline_misses << '/'
                    << options.rounds << ", dropped_releases=" << timing.dropped_releases
                    << ", max_backlog=" << timing.max_backlog_releases
                    << " (valid only for this backend benchmark boundary)\n";
        } else {
            std::cout << "Throughput mode has no release schedule and makes no deadline claim.\n";
        }
        if (options.rounds < 100000) {
            std::cout << "P99.99 not reported: use at least 100000 rounds with --csv.\n";
        }

        WriteCsv(options.csv_path,
            policy_name,
            loaded.exec_cfg.model_path,
            options,
            affinity_before_init,
            affinity_after_init,
            runtime,
            timing,
            backend_run_ms,
            ReadActionOutput(&ort, loaded));
        if (!options.csv_path.empty()) {
            std::cout << "Raw CSV: " << options.csv_path << "\n";
        }
        if (options.verbose_after_timing)
            PrintVerbose(timing, backend_run_ms);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "[benchmark_onnx_infer] " << error.what() << '\n';
        return 2;
    }
}
