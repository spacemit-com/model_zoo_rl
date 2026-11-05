/**
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file onnx_infer.cpp
 * @brief ONNX Runtime inference backend implementation
 */

#include "onnx_infer.h"

#include <onnxruntime_cxx_api.h>
#include <sched.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(cpu_rv64) || defined(__riscv)
#include "spacemit_ort_env.h"
#endif

namespace onnx_runtime {
namespace {

static_assert(
    static_cast<int>(TensorElementType::FLOAT32) == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
static_assert(
    static_cast<int>(TensorElementType::BFLOAT16) == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);

struct TensorTypeTraits {
    const char *name;
    std::size_t bits_per_element;
    bool supports_float_conversion;
};

constexpr std::array<TensorTypeTraits, 23> kTensorTypeTraits = {{
    {"undefined", 0, false},
    {"float32", 32, true},
    {"uint8", 8, true},
    {"int8", 8, true},
    {"uint16", 16, true},
    {"int16", 16, true},
    {"int32", 32, true},
    {"int64", 64, true},
    {"string", 0, false},
    {"bool", 8, true},
    {"float16", 16, true},
    {"float64", 64, true},
    {"uint32", 32, true},
    {"uint64", 64, true},
    {"complex64", 64, false},
    {"complex128", 128, false},
    {"bfloat16", 16, true},
    {"float8e4m3fn", 8, false},
    {"float8e4m3fnuz", 8, false},
    {"float8e5m2", 8, false},
    {"float8e5m2fnuz", 8, false},
    {"uint4", 4, false},
    {"int4", 4, false},
}};

static_assert(
    kTensorTypeTraits.size() == static_cast<std::size_t>(TensorElementType::INT4) + 1);

const TensorTypeTraits &GetTensorTypeTraits(TensorElementType type) {
    const int index = static_cast<int>(type);
    if (index < 0 || static_cast<std::size_t>(index) >= kTensorTypeTraits.size()) {
        return kTensorTypeTraits.front();
    }
    return kTensorTypeTraits[static_cast<std::size_t>(index)];
}

TensorElementType FromOnnxType(ONNXTensorElementDataType type) {
    return static_cast<TensorElementType>(static_cast<int>(type));
}

ONNXTensorElementDataType ToOnnxType(TensorElementType type) {
    return static_cast<ONNXTensorElementDataType>(static_cast<int>(type));
}

const char *TensorElementTypeName(TensorElementType type) {
    return GetTensorTypeTraits(type).name;
}

bool SupportsFloatConversion(TensorElementType type) {
    return GetTensorTypeTraits(type).supports_float_conversion;
}

std::size_t TensorByteCount(TensorElementType type, std::size_t element_count) {
    const auto &traits = GetTensorTypeTraits(type);
    if (traits.bits_per_element == 0) {
        throw std::runtime_error(
            std::string("unsupported tensor dtype: ") + traits.name);
    }
    if (traits.bits_per_element == 4) {
        return (element_count + 1) / 2;
    }
    return element_count * (traits.bits_per_element / 8);
}

template <typename T>
T ConvertInteger(float value, const std::string &tensor_name, std::size_t index) {
    if (!std::isfinite(value)) {
        throw std::runtime_error(
            "tensor " + tensor_name + " contains non-finite integer value at index " +
            std::to_string(index));
    }
    const long double rounded = std::round(static_cast<long double>(value));
    if (rounded < static_cast<long double>(std::numeric_limits<T>::lowest()) ||
        rounded > static_cast<long double>(std::numeric_limits<T>::max())) {
        throw std::runtime_error(
            "tensor " + tensor_name + " integer value out of range at index " +
            std::to_string(index));
    }
    return static_cast<T>(rounded);
}

template <typename T>
void AssignIntegers(
    const float *source, T *target, std::size_t count, const std::string &tensor_name) {
    for (std::size_t i = 0; i < count; ++i) {
        target[i] = ConvertInteger<T>(source[i], tensor_name, i);
    }
}

template <typename T>
void CopyNumbersToFloat(const T *source, std::size_t count, std::vector<float> &target) {
    target.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        target[i] = static_cast<float>(source[i]);
    }
}

void AssignFromFloat(
    Ort::Value &tensor,
    TensorElementType type,
    const float *source,
    std::size_t element_count,
    const std::string &tensor_name) {
    if (!source) {
        throw std::runtime_error("tensor " + tensor_name + " data is null");
    }
    void *raw = tensor.GetTensorMutableRawData();
    switch (type) {
    case TensorElementType::FLOAT32:
        std::memcpy(raw, source, element_count * sizeof(float));
        return;
    case TensorElementType::FLOAT64: {
        auto *target = static_cast<double *>(raw);
        for (std::size_t i = 0; i < element_count; ++i) target[i] = source[i];
        return;
    }
    case TensorElementType::UINT8:
        AssignIntegers(source, static_cast<std::uint8_t *>(raw), element_count, tensor_name);
        return;
    case TensorElementType::INT8:
        AssignIntegers(source, static_cast<std::int8_t *>(raw), element_count, tensor_name);
        return;
    case TensorElementType::UINT16:
        AssignIntegers(source, static_cast<std::uint16_t *>(raw), element_count, tensor_name);
        return;
    case TensorElementType::INT16:
        AssignIntegers(source, static_cast<std::int16_t *>(raw), element_count, tensor_name);
        return;
    case TensorElementType::UINT32:
        AssignIntegers(source, static_cast<std::uint32_t *>(raw), element_count, tensor_name);
        return;
    case TensorElementType::INT32:
        AssignIntegers(source, static_cast<std::int32_t *>(raw), element_count, tensor_name);
        return;
    case TensorElementType::UINT64:
        AssignIntegers(source, static_cast<std::uint64_t *>(raw), element_count, tensor_name);
        return;
    case TensorElementType::INT64:
        AssignIntegers(source, static_cast<std::int64_t *>(raw), element_count, tensor_name);
        return;
    case TensorElementType::BOOL: {
        auto *target = static_cast<bool *>(raw);
        for (std::size_t i = 0; i < element_count; ++i) {
            if (!std::isfinite(source[i])) {
                throw std::runtime_error(
                    "tensor " + tensor_name + " contains non-finite bool value at index " +
                    std::to_string(i));
            }
            target[i] = source[i] != 0.0f;
        }
        return;
    }
    case TensorElementType::FLOAT16: {
        auto *target = static_cast<std::uint16_t *>(raw);
        for (std::size_t i = 0; i < element_count; ++i) {
            target[i] = Ort::Float16_t(source[i]).val;
        }
        return;
    }
    case TensorElementType::BFLOAT16: {
        auto *target = static_cast<std::uint16_t *>(raw);
        for (std::size_t i = 0; i < element_count; ++i) {
            target[i] = Ort::BFloat16_t(source[i]).val;
        }
        return;
    }
    default:
        throw std::runtime_error(
            "tensor " + tensor_name + " requires native " +
            TensorElementTypeName(type) + " input data");
    }
}

void CopyToFloat(
    const Ort::Value &tensor,
    TensorElementType type,
    std::size_t element_count,
    std::vector<float> &target,
    const std::string &tensor_name) {
    const void *raw = tensor.GetTensorRawData();
    switch (type) {
    case TensorElementType::FLOAT32:
        target.resize(element_count);
        std::memcpy(target.data(), raw, element_count * sizeof(float));
        return;
    case TensorElementType::FLOAT64:
        CopyNumbersToFloat(
            static_cast<const double *>(raw), element_count, target);
        return;
    case TensorElementType::UINT8:
        CopyNumbersToFloat(
            static_cast<const std::uint8_t *>(raw), element_count, target);
        return;
    case TensorElementType::INT8:
        CopyNumbersToFloat(
            static_cast<const std::int8_t *>(raw), element_count, target);
        return;
    case TensorElementType::UINT16:
        CopyNumbersToFloat(
            static_cast<const std::uint16_t *>(raw), element_count, target);
        return;
    case TensorElementType::INT16:
        CopyNumbersToFloat(
            static_cast<const std::int16_t *>(raw), element_count, target);
        return;
    case TensorElementType::UINT32:
        CopyNumbersToFloat(
            static_cast<const std::uint32_t *>(raw), element_count, target);
        return;
    case TensorElementType::INT32:
        CopyNumbersToFloat(
            static_cast<const std::int32_t *>(raw), element_count, target);
        return;
    case TensorElementType::UINT64:
        CopyNumbersToFloat(
            static_cast<const std::uint64_t *>(raw), element_count, target);
        return;
    case TensorElementType::INT64:
        CopyNumbersToFloat(
            static_cast<const std::int64_t *>(raw), element_count, target);
        return;
    case TensorElementType::BOOL: {
        target.resize(element_count);
        const auto *source = static_cast<const bool *>(raw);
        for (std::size_t i = 0; i < element_count; ++i) {
            target[i] = source[i] ? 1.0f : 0.0f;
        }
        return;
    }
    case TensorElementType::FLOAT16: {
        target.resize(element_count);
        const auto *source = static_cast<const std::uint16_t *>(raw);
        for (std::size_t i = 0; i < element_count; ++i) {
            target[i] = Ort::Float16_t::FromBits(source[i]).ToFloat();
        }
        return;
    }
    case TensorElementType::BFLOAT16: {
        target.resize(element_count);
        const auto *source = static_cast<const std::uint16_t *>(raw);
        for (std::size_t i = 0; i < element_count; ++i) {
            target[i] = Ort::BFloat16_t::FromBits(source[i]).ToFloat();
        }
        return;
    }
    default:
        throw std::runtime_error(
            "tensor " + tensor_name + " cannot be represented as float: " +
            TensorElementTypeName(type));
    }
}

std::vector<int64_t> ResolveShape(const std::vector<int64_t> &shape) {
    std::vector<int64_t> resolved = shape;
    for (auto &dim : resolved) {
        if (dim <= 0) dim = 1;
    }
    return resolved;
}

int64_t ComputeTensorSize(const std::vector<int64_t> &shape) {
    int64_t size = 1;
    for (const auto dim : shape) size *= dim;
    return size;
}

Ort::Value CreateOwnedTensor(
    Ort::AllocatorWithDefaultOptions &allocator, const TensorInfo &info) {
    TensorByteCount(info.element_type, static_cast<std::size_t>(info.total_size));
    return Ort::Value::CreateTensor(
        allocator,
        info.shape.data(),
        info.shape.size(),
        ToOnnxType(info.element_type));
}

void ZeroTensor(Ort::Value &tensor, const TensorInfo &info) {
    const auto byte_count =
        TensorByteCount(info.element_type, static_cast<std::size_t>(info.total_size));
    if (byte_count > 0) {
        std::memset(tensor.GetTensorMutableRawData(), 0, byte_count);
    }
}

// CPU provider: one group applies to every worker; otherwise one group per
// worker (threads - 1, the caller thread is managed by the application).
// Groups use Linux CPU numbers like robot_base thread settings; ONNX Runtime
// numbers logical processors from 1, so each id is shifted by one.
std::string CpuWorkerAffinity(const std::string &affinity, int threads,
    std::set<int> *cpus) {
    if (affinity.empty() || threads <= 1) return {};
    const auto split = [](const std::string &text, char separator) {
        std::vector<std::string> parts;
        std::size_t begin = 0;
        while (true) {
            const std::size_t end = text.find(separator, begin);
            parts.push_back(text.substr(begin, end - begin));
            if (end == std::string::npos) return parts;
            begin = end + 1;
        }
    };
    const auto parse_cpu = [&affinity](const std::string &text) {
        if (text.empty() || text.size() > 4 ||
            text.find_first_not_of("0123456789") != std::string::npos) {
            throw std::runtime_error("invalid runtime.affinity: " + affinity);
        }
        const int cpu = std::stoi(text);
        if (cpu >= CPU_SETSIZE) throw std::runtime_error("invalid runtime.affinity: " + affinity);
        return cpu;
    };
    std::vector<std::string> groups;
    for (const auto &group : split(affinity, ';')) {
        std::string converted;
        for (const auto &item : split(group, ',')) {
            const auto range = split(item, '-');
            if (range.size() > 2) throw std::runtime_error("invalid runtime.affinity: " + affinity);
            const int first = parse_cpu(range.front());
            const int last = parse_cpu(range.back());
            if (last < first) throw std::runtime_error("invalid runtime.affinity: " + affinity);
            for (int cpu = first; cpu <= last; ++cpu) cpus->insert(cpu);
            if (!converted.empty()) converted += ',';
            converted += std::to_string(first + 1);
            if (range.size() == 2) converted += '-' + std::to_string(last + 1);
        }
        groups.push_back(converted);
    }
    const std::size_t workers = static_cast<std::size_t>(threads - 1);
    if (groups.size() == 1) groups.assign(workers, groups.front());
    if (groups.size() != workers) {
        throw std::runtime_error("runtime.affinity needs one group or threads-1 groups for "
            "the CPU provider: " + affinity);
    }
    std::string result;
    for (std::size_t i = 0; i < groups.size(); ++i) {
        if (i) result += ';';
        result += groups[i];
    }
    return result;
}

std::set<int> ProcessThreadIds() {
    std::set<int> threads;
    std::error_code error;
    for (std::filesystem::directory_iterator it("/proc/self/task", error), end;
        !error && it != end; it.increment(error)) {
        threads.insert(std::atoi(it->path().filename().c_str()));
    }
    return threads;
}

bool ThreadCpus(int tid, cpu_set_t *cpus) {
    CPU_ZERO(cpus);
    return sched_getaffinity(tid, sizeof(*cpus), cpus) == 0;
}

// "tid@cpus/policy" for threads that did not exist before session creation.
// ONNX Runtime workers move to configured CPUs once they start running, so
// wait briefly for them before reporting.
std::string DescribeNewThreads(const std::set<int> &before, const std::set<int> &requested) {
    std::vector<int> threads;
    for (const int tid : ProcessThreadIds()) {
        if (!before.count(tid)) threads.push_back(tid);
    }
    const auto settled = [&threads, &requested]() {
        for (const int tid : threads) {
            cpu_set_t cpus;
            if (!ThreadCpus(tid, &cpus)) continue;
            for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
                if (CPU_ISSET(cpu, &cpus) && !requested.count(cpu)) return false;
            }
        }
        return true;
    };
    if (!requested.empty()) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
        while (!settled() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    std::ostringstream text;
    for (const int tid : threads) {
        cpu_set_t cpus;
        if (!ThreadCpus(tid, &cpus)) continue;
        std::string list;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (!CPU_ISSET(cpu, &cpus)) continue;
            if (!list.empty()) list += ',';
            list += std::to_string(cpu);
        }
        const int policy = sched_getscheduler(tid);
        const char *name = policy == SCHED_FIFO ? "fifo" : policy == SCHED_RR ? "rr" :
            policy == SCHED_BATCH ? "batch" : policy == SCHED_IDLE ? "idle" : "other";
        if (text.tellp() > 0) text << ' ';
        text << tid << '@' << list << '/' << name;
    }
    return text.str();
}

}  // namespace

class OnnxRuntimeClass::ImpClass {
public:
    explicit ImpClass(OnnxRuntimeClass *omp);
    ~ImpClass();
    bool Init(const std::string &model_file, const RuntimeOptions &options);
    bool Step();
    void RequestTerminate();
    void ClearTerminate();

    OnnxRuntimeClass &omp;
    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    Ort::RunOptions run_options;
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<Ort::Value> inputs;
    std::vector<Ort::Value> outputs;
};

OnnxRuntimeClass::ImpClass::ImpClass(OnnxRuntimeClass *omp)
    : omp(*omp), env(ORT_LOGGING_LEVEL_WARNING, "OnnxRuntime") {
    std::cout << "[ONNX Runtime] ImpClass 构造" << std::endl;
}

OnnxRuntimeClass::ImpClass::~ImpClass() {
    std::cout << "[ONNX Runtime] ImpClass 析构" << std::endl;
}

bool OnnxRuntimeClass::ImpClass::Init(
    const std::string &model_file, const RuntimeOptions &options) {
    std::cout << "[ONNX Runtime] 开始初始化模型: " << model_file << std::endl;

    try {
        run_options.UnsetTerminate();
        if (options.provider != "auto" && options.provider != "cpu" &&
            options.provider != "spacemit") {
            throw std::runtime_error(
                "unsupported provider: " + options.provider +
                " (expected auto|cpu|spacemit)");
        }
        if (options.threads <= 0) {
            throw std::runtime_error("threads must be greater than zero");
        }

        omp.runtime_info_ = {};
        omp.runtime_info_.requested_provider = options.provider;
        omp.runtime_info_.affinity = options.affinity;
        omp.runtime_info_.ort_inter_threads = 1;
        omp.runtime_info_.ort_spinning = options.ort_spinning;

        const auto make_options = [&options] {
            auto value = std::make_unique<Ort::SessionOptions>();
            value->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            value->SetInterOpNumThreads(1);
            value->AddConfigEntry(
                "session.intra_op.allow_spinning",
                options.ort_spinning ? "1" : "0");
            return value;
        };
        auto next_options = make_options();
        // Only CPU sessions interpret affinity as worker CPU groups; the EP
        // receives the configured string unchanged.
        std::set<int> affinity_cpus;
        bool cpu_affinity_applied = false;
        const auto apply_cpu_affinity = [&](Ort::SessionOptions &value) {
            const std::string cpu_affinity =
                CpuWorkerAffinity(options.affinity, options.threads, &affinity_cpus);
            if (!cpu_affinity.empty()) {
                value.AddConfigEntry("session.intra_op_thread_affinities", cpu_affinity.c_str());
                cpu_affinity_applied = true;
            }
        };

        bool request_spacemit = options.provider == "spacemit";
#if defined(cpu_rv64) || defined(__riscv)
        request_spacemit = request_spacemit || options.provider == "auto";
#endif

        if (request_spacemit) {
#if defined(cpu_rv64) || defined(__riscv)
            next_options->SetIntraOpNumThreads(1);
            std::unordered_map<std::string, std::string> provider_options = {
                {"SPACEMIT_EP_INTRA_THREAD_NUM", std::to_string(options.threads)},
            };
            if (!options.affinity.empty()) {
                provider_options["SPACEMIT_EP_INTRA_THREAD_AFFINITY"] = options.affinity;
            }
            if (options.ep_dump_subgraphs) {
                provider_options["SPACEMIT_EP_DUMP_SUBGRAPHS"] = "1";
            }
            if (!options.ep_profile_prefix.empty()) {
                provider_options["SPACEMIT_EP_DEBUG_PROFILE"] = options.ep_profile_prefix;
            }
            const Ort::Status status =
                Ort::SessionOptionsSpaceMITEnvInit(*next_options, provider_options);
            if (!status.IsOK()) {
                const std::string error = status.GetErrorMessage();
                if (options.provider == "spacemit") {
                    throw std::runtime_error("SpaceMIT EP init failed: " + error);
                }
                std::cerr << "[ONNX Runtime] SpaceMIT EP 初始化失败，auto 回退 CPU: "
                    << error << std::endl;
                next_options = make_options();
                next_options->SetIntraOpNumThreads(options.threads);
                std::string affinity_note;
                try {
                    apply_cpu_affinity(*next_options);
                } catch (const std::exception &affinity_error) {
                    affinity_cpus.clear();
                    affinity_note = std::string("; CPU worker affinity ignored: ") + affinity_error.what();
                }
                omp.runtime_info_.initialized_provider = "cpu";
                omp.runtime_info_.ort_intra_threads = options.threads;
                omp.runtime_info_.provider_status =
                    "SpaceMIT EP init failed; auto fallback to CPU: " + error + affinity_note;
            } else {
                omp.runtime_info_.initialized_provider = "spacemit-registered";
                omp.runtime_info_.ort_intra_threads = 1;
                omp.runtime_info_.ep_threads = options.threads;
                omp.runtime_info_.provider_status =
                    "SpaceMIT EP init OK; node assignment is not verified and CPU fallback is allowed";
            }
#else
            throw std::runtime_error(
                "SpaceMIT provider requested by a non-RISC-V build");
#endif
        } else {
            next_options->SetIntraOpNumThreads(options.threads);
            apply_cpu_affinity(*next_options);
            omp.runtime_info_.initialized_provider = "cpu";
            omp.runtime_info_.ort_intra_threads = options.threads;
            omp.runtime_info_.provider_status = "CPU provider selected";
        }

        std::cout << "[ONNX Runtime] provider requested="
            << omp.runtime_info_.requested_provider
            << ", initialized=" << omp.runtime_info_.initialized_provider
            << ", ORT intra=" << omp.runtime_info_.ort_intra_threads
            << ", ORT inter=" << omp.runtime_info_.ort_inter_threads
            << ", EP threads requested=" << omp.runtime_info_.ep_threads
            << ", ORT spinning="
            << (omp.runtime_info_.ort_spinning ? "on" : "off")
            << ", affinity requested="
            << (omp.runtime_info_.affinity.empty()
                    ? "not-set" : omp.runtime_info_.affinity)
            << std::endl;

        const auto threads_before = ProcessThreadIds();
        auto next_session =
            std::make_unique<Ort::Session>(env, model_file.c_str(), *next_options);
        omp.runtime_info_.worker_threads = DescribeNewThreads(threads_before,
            cpu_affinity_applied ? affinity_cpus : std::set<int>{});
        std::cout << "[ONNX Runtime] 模型加载成功, worker threads="
            << (omp.runtime_info_.worker_threads.empty() ? "none" : omp.runtime_info_.worker_threads)
            << std::endl;

        const std::size_t input_count = next_session->GetInputCount();
        std::cout << "[ONNX Runtime] 检测到 " << input_count << " 个输入" << std::endl;
        std::vector<TensorInfo> next_input_infos(input_count);
        std::vector<Ort::Value> next_inputs;
        next_inputs.reserve(input_count);

        for (std::size_t i = 0; i < input_count; ++i) {
            auto input_name = next_session->GetInputNameAllocated(i, allocator);
            auto type_info = next_session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto &info = next_input_infos[i];
            info.name = input_name.get();
            info.shape = ResolveShape(tensor_info.GetShape());
            info.total_size = ComputeTensorSize(info.shape);
            info.element_type = FromOnnxType(tensor_info.GetElementType());
            info.element_type_name = TensorElementTypeName(info.element_type);
            next_inputs.push_back(CreateOwnedTensor(allocator, info));
            ZeroTensor(next_inputs.back(), info);

            std::cout << "  输入[" << i << "]: " << info.name << ", shape=[";
            for (std::size_t j = 0; j < info.shape.size(); ++j) {
                std::cout << info.shape[j];
                if (j + 1 < info.shape.size()) std::cout << ", ";
            }
            std::cout << "], dtype=" << info.element_type_name
                    << ", total_size=" << info.total_size << std::endl;
        }

        const std::size_t output_count = next_session->GetOutputCount();
        std::cout << "[ONNX Runtime] 检测到 " << output_count << " 个输出" << std::endl;
        std::vector<TensorInfo> next_output_infos(output_count);

        for (std::size_t i = 0; i < output_count; ++i) {
            auto output_name = next_session->GetOutputNameAllocated(i, allocator);
            auto type_info = next_session->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto &info = next_output_infos[i];
            info.name = output_name.get();
            info.shape = ResolveShape(tensor_info.GetShape());
            info.total_size = ComputeTensorSize(info.shape);
            info.element_type = FromOnnxType(tensor_info.GetElementType());
            info.element_type_name = TensorElementTypeName(info.element_type);

            std::cout << "  输出[" << i << "]: " << info.name << ", shape=[";
            for (std::size_t j = 0; j < info.shape.size(); ++j) {
                std::cout << info.shape[j];
                if (j + 1 < info.shape.size()) std::cout << ", ";
            }
            std::cout << "], dtype=" << info.element_type_name
                    << ", total_size=" << info.total_size << std::endl;
        }

        session = std::move(next_session);
        inputs = std::move(next_inputs);
        outputs.clear();
        omp.input_infos_ = std::move(next_input_infos);
        omp.output_infos_ = std::move(next_output_infos);
        omp.output_float_views_.clear();
        omp.output_float_views_.resize(output_count);
        omp.last_error_.clear();
        std::cout << "[ONNX Runtime] 初始化完成" << std::endl;
        return true;
    } catch (const Ort::Exception &e) {
        omp.last_error_ = e.what();
        std::cerr << "[ONNX Runtime] 初始化错误: " << e.what() << std::endl;
        return false;
    } catch (const std::exception &e) {
        omp.last_error_ = e.what();
        std::cerr << "[ONNX Runtime] 初始化异常: " << e.what() << std::endl;
        return false;
    }
}

bool OnnxRuntimeClass::ImpClass::Step() {
    omp.outputs_valid_ = false;
    if (!session) {
        omp.last_error_ = "会话未初始化";
        std::cerr << "[ONNX Runtime] 错误: " << omp.last_error_ << std::endl;
        return false;
    }

    try {
        std::vector<const char *> input_names;
        input_names.reserve(omp.input_infos_.size());
        for (const auto &info : omp.input_infos_) input_names.push_back(info.name.c_str());

        std::vector<const char *> output_names;
        output_names.reserve(omp.output_infos_.size());
        for (const auto &info : omp.output_infos_) output_names.push_back(info.name.c_str());

        auto next_outputs = session->Run(
            run_options,
            input_names.data(),
            inputs.data(),
            inputs.size(),
            output_names.data(),
            output_names.size());

        if (next_outputs.size() != omp.output_infos_.size()) {
            throw std::runtime_error("ONNX output count changed at runtime");
        }
        for (std::size_t i = 0; i < next_outputs.size(); ++i) {
            if (!next_outputs[i].IsTensor()) {
                throw std::runtime_error(
                    "output " + omp.output_infos_[i].name + " is not a tensor");
            }
            const auto actual_info = next_outputs[i].GetTensorTypeAndShapeInfo();
            const auto actual_type = FromOnnxType(actual_info.GetElementType());
            const auto &expected = omp.output_infos_[i];
            if (actual_type != expected.element_type) {
                throw std::runtime_error(
                    "tensor " + expected.name + " output dtype changed from " +
                    expected.element_type_name + " to " + TensorElementTypeName(actual_type));
            }
            if (actual_info.GetElementCount() != static_cast<std::size_t>(expected.total_size)) {
                throw std::runtime_error(
                    "tensor " + expected.name + " output size changed at runtime");
            }
        }

        outputs = std::move(next_outputs);
        omp.outputs_valid_ = true;
        omp.last_error_.clear();
        return true;
    } catch (const Ort::Exception &e) {
        omp.last_error_ = e.what();
        std::cerr << "[ONNX Runtime] 推理错误: " << e.what() << std::endl;
    } catch (const std::exception &e) {
        omp.last_error_ = e.what();
        std::cerr << "[ONNX Runtime] 推理异常: " << e.what() << std::endl;
    }

    return false;
}

void OnnxRuntimeClass::ImpClass::RequestTerminate() {
    run_options.SetTerminate();
}

void OnnxRuntimeClass::ImpClass::ClearTerminate() {
    run_options.UnsetTerminate();
}

OnnxRuntimeClass::OnnxRuntimeClass() : imp_(std::make_unique<ImpClass>(this)) {
    std::cout << "[ONNX Runtime] OnnxRuntimeClass 构造" << std::endl;
}

OnnxRuntimeClass::~OnnxRuntimeClass() {
    std::cout << "[ONNX Runtime] OnnxRuntimeClass 析构" << std::endl;
}

bool OnnxRuntimeClass::Init(const std::string &model_file) {
    return Init(model_file, RuntimeOptions{});
}

bool OnnxRuntimeClass::Init(
    const std::string &model_file, const RuntimeOptions &options) {
    outputs_valid_ = false;
    if (!imp_) {
        last_error_ = "ImpClass 未初始化";
        std::cerr << "[ONNX Runtime] 错误: " << last_error_ << std::endl;
        return false;
    }
    return imp_->Init(model_file, options);
}

bool OnnxRuntimeClass::Run() {
    if (!imp_) {
        last_error_ = "ImpClass 未初始化";
        std::cerr << "[ONNX Runtime] 错误: " << last_error_ << std::endl;
        return false;
    }
    return imp_->Step();
}

void OnnxRuntimeClass::RequestTerminate() {
    if (imp_) imp_->RequestTerminate();
}

void OnnxRuntimeClass::ClearTerminate() {
    if (imp_) imp_->ClearTerminate();
}

const std::string &OnnxRuntimeClass::GetLastError() const {
    return last_error_;
}

const RuntimeInfo &OnnxRuntimeClass::GetRuntimeInfo() const {
    return runtime_info_;
}

void OnnxRuntimeClass::EnsureOutputsValid() const {
    if (!outputs_valid_) {
        throw std::runtime_error("ONNX outputs are unavailable until Run() succeeds");
    }
}

int OnnxRuntimeClass::GetInputCount() const {
    return static_cast<int>(input_infos_.size());
}

int OnnxRuntimeClass::GetOutputCount() const {
    return static_cast<int>(output_infos_.size());
}

const TensorInfo &OnnxRuntimeClass::GetInputInfo(int index) const {
    return input_infos_.at(index);
}

const TensorInfo &OnnxRuntimeClass::GetOutputInfo(int index) const {
    return output_infos_.at(index);
}

void OnnxRuntimeClass::SetInputFromFloat(
    int index, const float *data, std::size_t element_count) {
    const auto &info = input_infos_.at(index);
    if (element_count != static_cast<std::size_t>(info.total_size)) {
        throw std::runtime_error("tensor " + info.name + " input size mismatch");
    }
    AssignFromFloat(
        imp_->inputs.at(index), info.element_type, data, element_count, info.name);
}

bool OnnxRuntimeClass::CanSetInputFromFloat(int index) const {
    return SupportsFloatConversion(input_infos_.at(index).element_type);
}

void OnnxRuntimeClass::SetInput(int index, const TensorView &input) {
    const auto &info = input_infos_.at(index);
    const auto expected_count = static_cast<std::size_t>(info.total_size);
    const auto expected_bytes = TensorByteCount(info.element_type, expected_count);
    if (input.element_type != info.element_type) {
        throw std::runtime_error(
            "tensor " + info.name + " dtype mismatch: actual=" +
            TensorElementTypeName(input.element_type) + ", expected=" +
            info.element_type_name);
    }
    if (input.element_count != expected_count || input.byte_count != expected_bytes) {
        throw std::runtime_error("tensor " + info.name + " buffer size mismatch");
    }
    if (expected_bytes > 0 && !input.data) {
        throw std::runtime_error("tensor " + info.name + " data is null");
    }
    if (expected_bytes > 0) {
        std::memcpy(
            imp_->inputs.at(index).GetTensorMutableRawData(), input.data, expected_bytes);
    }
}

const std::vector<float> &OnnxRuntimeClass::GetOutput(int index) const {
    EnsureOutputsValid();
    const auto &info = output_infos_.at(index);
    auto &target = output_float_views_.at(index);
    CopyToFloat(
        imp_->outputs.at(index),
        info.element_type,
        static_cast<std::size_t>(info.total_size),
        target,
        info.name);
    return target;
}

bool OnnxRuntimeClass::CanGetOutputAsFloat(int index) const {
    return SupportsFloatConversion(output_infos_.at(index).element_type);
}

TensorView OnnxRuntimeClass::GetOutputView(int index) const {
    EnsureOutputsValid();
    const auto &info = output_infos_.at(index);
    const auto element_count = static_cast<std::size_t>(info.total_size);
    return {
        info.element_type,
        imp_->outputs.at(index).GetTensorRawData(),
        element_count,
        TensorByteCount(info.element_type, element_count),
    };
}

void OnnxRuntimeClass::CopyOutputToInput(int output_index, int input_index) {
    EnsureOutputsValid();
    const auto &output_info = output_infos_.at(output_index);
    const auto &input_info = input_infos_.at(input_index);
    if (output_info.element_type != input_info.element_type ||
        output_info.total_size != input_info.total_size) {
        throw std::runtime_error(
            "feedback tensor mismatch: " + input_info.name + " <- " + output_info.name);
    }
    const auto byte_count = TensorByteCount(
        input_info.element_type, static_cast<std::size_t>(input_info.total_size));
    if (byte_count > 0) {
        std::memcpy(
            imp_->inputs.at(input_index).GetTensorMutableRawData(),
            imp_->outputs.at(output_index).GetTensorRawData(),
            byte_count);
    }
}

void OnnxRuntimeClass::PrintModelInfo() const {
    std::cout << "\n========== ONNX 模型信息 ==========" << std::endl;
    std::cout << "输入数量: " << GetInputCount() << std::endl;
    for (int i = 0; i < GetInputCount(); ++i) {
        const auto &info = input_infos_[i];
        std::cout << "  [" << i << "] " << info.name << ": [";
        for (std::size_t j = 0; j < info.shape.size(); ++j) {
            std::cout << info.shape[j];
            if (j + 1 < info.shape.size()) std::cout << ", ";
        }
        std::cout << "] " << info.element_type_name
                << " (total: " << info.total_size << ")" << std::endl;
    }

    std::cout << "输出数量: " << GetOutputCount() << std::endl;
    for (int i = 0; i < GetOutputCount(); ++i) {
        const auto &info = output_infos_[i];
        std::cout << "  [" << i << "] " << info.name << ": [";
        for (std::size_t j = 0; j < info.shape.size(); ++j) {
            std::cout << info.shape[j];
            if (j + 1 < info.shape.size()) std::cout << ", ";
        }
        std::cout << "] " << info.element_type_name
                << " (total: " << info.total_size << ")" << std::endl;
    }
    std::cout << "===================================" << std::endl;
}

}  // namespace onnx_runtime
