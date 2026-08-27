/**
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file benchmark_policy_executor.cpp
 * @brief RL 组件内部 PolicyExecutor 链路与周期调度 benchmark
 *
 * 测量 AssembleObs -> Infer -> MapActionToTargetPos。输入由 YAML 配置和固定的
 * synthetic robot state 构造；本工具无法构造的 external input 或 custom array
 * 会在运行前被明确拒绝。
 */

#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "benchmark_common.h"
#include "rl_service.h"

namespace {

using rl_benchmark::Clock;
using rl_benchmark::Options;
using rl_benchmark::Stats;
using rl_benchmark::TimingResult;
using rl_policy::LoadedPolicyConfig;
using rl_policy::ModelInputSource;
using rl_policy::PolicyExecutor;

struct ComponentSamples {
    std::vector<double> obs_ms;
    std::vector<double> infer_ms;
    std::vector<double> map_ms;

    void Reserve(int rounds) {
        obs_ms.reserve(rounds);
        infer_ms.reserve(rounds);
        map_ms.reserve(rounds);
    }
};

void PrintUsage(const char *program) {
    std::cerr << "用法: " << program << " <yaml_path> <policy_name> <robot_dir> [options]\n";
    rl_benchmark::PrintCommonUsage(std::cerr);
}

void ValidatePipelineBoundary(const LoadedPolicyConfig &loaded) {
    std::vector<std::string> reasons;
    for (const auto &input : loaded.exec_cfg.model_io.inputs) {
        if (input.source != ModelInputSource::EXTERNAL ||
            !input.initial_value.empty()) {
            continue;
        }
        const std::string key = input.key.empty() ? input.name : input.key;
        reasons.push_back("external_without_default=" + key);
    }
    if (!loaded.exec_cfg.custom_array_dims.empty()) {
        reasons.push_back("custom_array observation injection");
    }
    if (reasons.empty())
        return;

    std::string detail;
    for (const auto &reason : reasons) {
        if (!detail.empty())
            detail += ", ";
        detail += reason;
    }
    throw std::runtime_error(
        "benchmark_policy_executor cannot populate required inputs (" + detail +
        "). Provide them in an integration or replay runner; use "
        "benchmark_onnx_infer only for explicitly synthetic backend timing.");
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
    double input_dt,
    const std::string &affinity_before_init,
    const std::string &affinity_after_init,
    const rl_policy::InferenceRuntimeInfo &runtime,
    const TimingResult &timing,
    const ComponentSamples &components) {
    if (path.empty())
        return;
    auto output = rl_benchmark::OpenCsv(path);
    output << "# benchmark=rl_policy_executor_component_pipeline\n"
        << "# policy=" << policy_name << "\n"
        << "# model=" << model_path << "\n"
        << "# build_type=" << RL_BENCHMARK_BUILD_TYPE << "\n"
        << "# cxx_flags=" << RL_BENCHMARK_CXX_FLAGS << "\n"
        << "# mode=" << rl_benchmark::ModeName(options.mode) << "\n"
        << "# hz=" << options.hz << "\n"
        << "# input_dt=" << input_dt << "\n"
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
        << "# ep_profile_prefix=" << options.ep_profile_prefix << "\n"
        << "# measure_start_delay_ms=" << options.measure_start_delay_ms << "\n"
        << "iteration,release_index,dropped_before,backlog_releases,"
            "obs_ms,infer_ms,map_ms,service_ms,release_jitter_ms,response_ms,"
            "deadline_lateness_ms\n";
    for (std::size_t i = 0; i < timing.samples.size(); ++i) {
        const auto &sample = timing.samples[i];
        output << i << ',' << sample.release_index << ',' << sample.dropped_before << ','
            << sample.backlog_releases << ',' << components.obs_ms[i] << ','
            << components.infer_ms[i] << ',' << components.map_ms[i] << ',' << sample.service_ms
            << ',' << sample.release_jitter_ms << ',' << sample.response_ms << ','
            << sample.deadline_lateness_ms << '\n';
    }
}

void PrintVerbose(const TimingResult &timing, const ComponentSamples &components) {
    for (std::size_t i = 0; i < timing.samples.size(); ++i) {
        const auto &sample = timing.samples[i];
        std::cout << "sample=" << i << " release=" << sample.release_index
                << " obs_ms=" << components.obs_ms[i] << " infer_ms=" << components.infer_ms[i]
                << " map_ms=" << components.map_ms[i] << " service_ms=" << sample.service_ms;
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
        LoadedPolicyConfig loaded
            = rl_policy::LoadPolicyConfigFromYaml(yaml_path, policy_name, robot_dir);
        ValidatePipelineBoundary(loaded);

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

        loaded.exec_cfg.runtime.provider = options.provider;
        loaded.exec_cfg.runtime.threads = options.threads;
        loaded.exec_cfg.runtime.affinity = ep_affinity;
        loaded.exec_cfg.runtime.ep_dump_subgraphs = options.ep_dump_subgraphs;
        loaded.exec_cfg.runtime.ep_profile_prefix = options.ep_profile_prefix;
        loaded.exec_cfg.runtime.ort_spinning = options.ort_spinning;

        PolicyExecutor policy;
        policy.Init(loaded.exec_cfg);
        const std::string affinity_after_init = rl_benchmark::ApplyAndGetAffinity("");
        const auto runtime = policy.GetRuntimeInfo();
        const int num_dof = static_cast<int>(loaded.exec_cfg.rl_default_pos.size());

        std::cout << "\nRL PolicyExecutor component benchmark\n"
                << "Policy:      " << policy_name << "\n"
                << "Model:       " << loaded.exec_cfg.model_path << "\n"
                << "Semantics:   PolicyExecutor step with fixed synthetic robot state\n"
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
                << "Obs/action:  " << policy.ObsDim() << '/' << policy.ActionDim()
                << ", DOF=" << num_dof << "\n"
                << "Feedback:    " << policy.FeedbackStateCount()
                << ", obs history=" << (policy.HasObsHist() ? "yes" : "no") << "\n"
                << "Warmup/test: " << options.warmup << '/' << options.rounds << "\n";
        rl_benchmark::PrintBuildMetadata();
        std::cout << "Thread count and affinity are configuration evidence, not proof that "
                    "every allowed CPU stayed busy.\n";

        const std::array<double, 3> gyro = {0.01, -0.02, 0.03};
        const std::array<double, 3> rpy = {0.0, 0.0, 0.0};
        const std::array<double, 4> base_quat = {1.0, 0.0, 0.0, 0.0};
        const std::array<double, 3> base_vel = {0.0, 0.0, 0.0};
        const double cmd_vx = loaded.command_init[0];
        const double cmd_vy = loaded.command_init[1];
        const double cmd_wz = loaded.command_init[2];
        const std::vector<double> joint_pos = loaded.exec_cfg.rl_default_pos;
        const std::vector<double> joint_vel(num_dof, 0.0);

        Eigen::VectorXf obs;
        std::vector<double> action;
        std::vector<double> target_pos;
        ComponentSamples components;
        components.Reserve(options.rounds);

        const auto run_step = [&](bool measured) {
            const auto obs_start = Clock::now();
            policy.AssembleObs(gyro,
                rpy,
                cmd_vx,
                cmd_vy,
                cmd_wz,
                joint_pos,
                joint_vel,
                base_quat,
                base_vel,
                static_cast<float>(input_dt),
                obs);
            const auto infer_start = Clock::now();
            policy.Infer(obs, action);
            const auto map_start = Clock::now();
            policy.MapActionToTargetPos(action, target_pos);
            const auto finish = Clock::now();
            if (measured) {
                components.obs_ms.push_back(rl_benchmark::ToMilliseconds(infer_start - obs_start));
                components.infer_ms.push_back(
                    rl_benchmark::ToMilliseconds(map_start - infer_start));
                components.map_ms.push_back(rl_benchmark::ToMilliseconds(finish - map_start));
            }
        };

        std::cout << "Warmup...\n" << std::flush;
        for (int i = 0; i < options.warmup; ++i)
            run_step(false);

        rl_benchmark::BeginMeasuredRegion(options);
        const TimingResult timing
            = rl_benchmark::MeasureRounds(options, [&](std::uint64_t) { run_step(true); });
        rl_benchmark::EndMeasuredRegion();

        PrintStats("Obs assembly", rl_benchmark::ComputeStats(components.obs_ms));
        PrintStats("Inference", rl_benchmark::ComputeStats(components.infer_ms));
        PrintStats("Action mapping", rl_benchmark::ComputeStats(components.map_ms));
        PrintStats("RL component step",
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
                    << " (valid only for this PolicyExecutor benchmark boundary)\n";
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
            input_dt,
            affinity_before_init,
            affinity_after_init,
            runtime,
            timing,
            components);
        if (!options.csv_path.empty()) {
            std::cout << "Raw CSV: " << options.csv_path << "\n";
        }
        if (options.verbose_after_timing)
            PrintVerbose(timing, components);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "[benchmark_policy_executor] " << error.what() << '\n';
        return 2;
    }
}
