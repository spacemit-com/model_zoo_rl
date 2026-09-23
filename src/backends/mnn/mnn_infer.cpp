/**
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file mnn_infer.cpp
 * @brief MNN inference backend implementation
 */

#include "mnn_infer.h"

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mnn_runtime {
namespace {

struct TensorTypeTraits {
    const char *name;
    std::size_t bytes_per_element;
};

constexpr std::array<TensorTypeTraits, 14> kTensorTypeTraits = {{
    {"undefined", 0},
    {"float32", 4},
    {"uint8", 1},
    {"int8", 1},
    {"uint16", 2},
    {"int16", 2},
    {"int32", 4},
    {"int64", 8},
    {"string", 0},
    {"bool", 1},
    {"float16", 2},
    {"float64", 8},
    {"uint32", 4},
    {"uint64", 8},
}};

const TensorTypeTraits &GetTensorTypeTraits(TensorElementType type) {
    const int index = static_cast<int>(type);
    if (index < 0 || static_cast<std::size_t>(index) >= kTensorTypeTraits.size()) {
        return kTensorTypeTraits.front();
    }
    return kTensorTypeTraits[static_cast<std::size_t>(index)];
}

TensorElementType FromMnnType(const halide_type_t &type) {
    if (type.lanes != 1) {
        throw std::runtime_error(
            "unsupported MNN tensor lanes: " + std::to_string(type.lanes));
    }
    if (type.code == halide_type_float) {
        if (type.bits == 16) return TensorElementType::FLOAT16;
        if (type.bits == 32) return TensorElementType::FLOAT32;
        if (type.bits == 64) return TensorElementType::FLOAT64;
    } else if (type.code == halide_type_int) {
        if (type.bits == 8) return TensorElementType::INT8;
        if (type.bits == 16) return TensorElementType::INT16;
        if (type.bits == 32) return TensorElementType::INT32;
        if (type.bits == 64) return TensorElementType::INT64;
    } else if (type.code == halide_type_uint) {
        if (type.bits == 1) return TensorElementType::BOOL;
        if (type.bits == 8) return TensorElementType::UINT8;
        if (type.bits == 16) return TensorElementType::UINT16;
        if (type.bits == 32) return TensorElementType::UINT32;
        if (type.bits == 64) return TensorElementType::UINT64;
    }
    throw std::runtime_error(
        "unsupported MNN tensor dtype: code=" + std::to_string(type.code) +
        ", bits=" + std::to_string(type.bits));
}

std::size_t TensorByteCount(TensorElementType type, std::size_t element_count) {
    const auto &traits = GetTensorTypeTraits(type);
    if (traits.bytes_per_element == 0) {
        throw std::runtime_error(
            std::string("unsupported tensor dtype: ") + traits.name);
    }
    return element_count * traits.bytes_per_element;
}

}  // namespace

class MnnRuntimeClass::ImpClass {
public:
    std::unique_ptr<MNN::Interpreter> interpreter;
    MNN::BackendConfig backend_config;
    MNN::Session *session = nullptr;
    std::vector<MNN::Tensor *> input_tensors;
    std::vector<MNN::Tensor *> output_tensors;
    std::vector<std::unique_ptr<MNN::Tensor>> host_input_tensors;
    std::vector<std::unique_ptr<MNN::Tensor>> host_output_tensors;
    std::vector<std::vector<std::uint8_t>> input_buffers;
};

MnnRuntimeClass::MnnRuntimeClass() : imp_(std::make_unique<ImpClass>()) {}

MnnRuntimeClass::~MnnRuntimeClass() {
    if (imp_->interpreter && imp_->session) {
        imp_->interpreter->releaseSession(imp_->session);
    }
}

bool MnnRuntimeClass::Init(const std::string &model_file) {
    RuntimeOptions options;
    return Init(model_file, options);
}

bool MnnRuntimeClass::Init(const std::string &model_file, const RuntimeOptions &options) {
    last_error_.clear();
    outputs_valid_ = false;
    terminate_requested_.store(false, std::memory_order_release);

    try {
        if (options.backend != "cpu" && options.backend != "auto") {
            last_error_ = "MNN supports only cpu or auto backend";
            return false;
        }
        if (options.threads <= 0) {
            last_error_ = "MNN thread count must be positive";
            return false;
        }
        if (imp_->interpreter && imp_->session) {
            imp_->interpreter->releaseSession(imp_->session);
            imp_->session = nullptr;
        }
        imp_->interpreter.reset();
        input_infos_.clear();
        output_infos_.clear();
        output_float_views_.clear();
        imp_->input_tensors.clear();
        imp_->output_tensors.clear();
        imp_->host_input_tensors.clear();
        imp_->host_output_tensors.clear();
        imp_->input_buffers.clear();

        // 创建 interpreter
        imp_->interpreter = std::unique_ptr<MNN::Interpreter>(
            MNN::Interpreter::createFromFile(model_file.c_str()));
        if (!imp_->interpreter) {
            last_error_ = "failed to load MNN model: " + model_file;
            return false;
        }

        // 配置 session
        MNN::ScheduleConfig config;
        config.type = MNN_FORWARD_CPU;
        config.numThread = options.threads;
        imp_->backend_config.precision = options.high_precision
                                            ? MNN::BackendConfig::Precision_High
                                            : MNN::BackendConfig::Precision_Normal;
        config.backendConfig = &imp_->backend_config;

        imp_->session = imp_->interpreter->createSession(config);
        if (!imp_->session) {
            last_error_ = "failed to create MNN session";
            return false;
        }

        // 获取输入输出张量信息
        auto input_map = imp_->interpreter->getSessionInputAll(imp_->session);
        auto output_map = imp_->interpreter->getSessionOutputAll(imp_->session);

        // 处理输入张量
        for (const auto &[name, tensor] : input_map) {
            TensorInfo info;
            info.name = name;
            info.element_type = FromMnnType(tensor->getType());
            info.element_type_name = GetTensorTypeTraits(info.element_type).name;
            info.total_size = tensor->elementSize();

            // 获取形状
            for (int i = 0; i < tensor->dimensions(); ++i) {
                info.shape.push_back(tensor->length(i));
            }

            input_infos_.push_back(std::move(info));
            imp_->input_tensors.push_back(tensor);
            imp_->host_input_tensors.push_back(
                std::make_unique<MNN::Tensor>(tensor, MNN::Tensor::CAFFE));
            imp_->input_buffers.emplace_back(
                TensorByteCount(input_infos_.back().element_type, tensor->elementSize()));
        }

        // 处理输出张量
        for (const auto &[name, tensor] : output_map) {
            TensorInfo info;
            info.name = name;
            info.element_type = FromMnnType(tensor->getType());
            info.element_type_name = GetTensorTypeTraits(info.element_type).name;
            info.total_size = tensor->elementSize();

            // 获取形状
            for (int i = 0; i < tensor->dimensions(); ++i) {
                info.shape.push_back(tensor->length(i));
            }

            output_infos_.push_back(std::move(info));
            imp_->output_tensors.push_back(tensor);
            imp_->host_output_tensors.push_back(
                std::make_unique<MNN::Tensor>(tensor, MNN::Tensor::CAFFE));
        }

        output_float_views_.resize(output_infos_.size());

        // 填充 RuntimeInfo
        runtime_info_.requested_backend = options.backend;
        runtime_info_.initialized_backend = "cpu";
        runtime_info_.threads = options.threads;
        runtime_info_.high_precision = options.high_precision;
        runtime_info_.backend_status = "ok";

        return true;

    } catch (const std::exception &e) {
        last_error_ = std::string("MNN initialization error: ") + e.what();
        return false;
    }
}

bool MnnRuntimeClass::Run() {
    last_error_.clear();
    outputs_valid_ = false;

    if (!imp_->session || !imp_->interpreter) {
        last_error_ = "MNN session not initialized";
        return false;
    }

    try {
        // 复制输入数据到 MNN 张量
        for (size_t i = 0; i < imp_->input_tensors.size(); ++i) {
            if (!imp_->input_tensors[i]->copyFromHostTensor(imp_->host_input_tensors[i].get())) {
                last_error_ = "failed to copy input tensor " + std::to_string(i);
                return false;
            }
        }

        // 执行推理
        const auto continue_running = [this](
            const std::vector<MNN::Tensor *> &,
            const std::string &) {
            return !terminate_requested_.load(std::memory_order_acquire);
        };
        const auto result = imp_->interpreter->runSessionWithCallBack(
            imp_->session,
            [](const std::vector<MNN::Tensor *> &, const std::string &) { return true; },
            continue_running,
            true);
        if (result != MNN::NO_ERROR) {
            last_error_ = result == MNN::CALL_BACK_STOP &&
                    terminate_requested_.load(std::memory_order_acquire)
                ? "MNN inference terminated"
                : "MNN inference failed with error code " + std::to_string(result);
            return false;
        }

        // 读取输出数据
        for (size_t i = 0; i < imp_->output_tensors.size(); ++i) {
            if (!imp_->output_tensors[i]->copyToHostTensor(imp_->host_output_tensors[i].get())) {
                last_error_ = "failed to read output tensor " + std::to_string(i);
                return false;
            }
        }

        outputs_valid_ = true;
        return true;

    } catch (const std::exception &e) {
        last_error_ = std::string("MNN inference error: ") + e.what();
        return false;
    }
}

void MnnRuntimeClass::RequestTerminate() {
    terminate_requested_.store(true, std::memory_order_release);
}

const std::string &MnnRuntimeClass::GetLastError() const {
    return last_error_;
}

const RuntimeInfo &MnnRuntimeClass::GetRuntimeInfo() const {
    return runtime_info_;
}

int MnnRuntimeClass::GetInputCount() const {
    return static_cast<int>(input_infos_.size());
}

int MnnRuntimeClass::GetOutputCount() const {
    return static_cast<int>(output_infos_.size());
}

const TensorInfo &MnnRuntimeClass::GetInputInfo(int index) const {
    if (index < 0 || index >= GetInputCount()) {
        throw std::out_of_range("input index out of range");
    }
    return input_infos_[static_cast<std::size_t>(index)];
}

const TensorInfo &MnnRuntimeClass::GetOutputInfo(int index) const {
    if (index < 0 || index >= GetOutputCount()) {
        throw std::out_of_range("output index out of range");
    }
    return output_infos_[static_cast<std::size_t>(index)];
}

void MnnRuntimeClass::SetInputFromFloat(int index, const float *data, std::size_t element_count) {
    if (index < 0 || index >= GetInputCount()) {
        throw std::out_of_range("input index out of range");
    }
    if (!data) {
        throw std::invalid_argument("input data is null");
    }

    const auto &info = input_infos_[static_cast<std::size_t>(index)];
    if (info.element_type != TensorElementType::FLOAT32) {
        throw std::invalid_argument("input tensor " + info.name + " does not support float input");
    }
    if (element_count != static_cast<std::size_t>(info.total_size)) {
        throw std::invalid_argument(
            "input element count mismatch: expected " + std::to_string(info.total_size) +
            " got " + std::to_string(element_count));
    }

    // 验证数据有效性
    for (std::size_t i = 0; i < element_count; ++i) {
        if (!std::isfinite(data[i])) {
            throw std::invalid_argument(
                "input tensor " + info.name + " contains non-finite value at index " +
                std::to_string(i));
        }
    }

    // 复制到 host tensor
    float *host_data = imp_->host_input_tensors[static_cast<std::size_t>(index)]->host<float>();
    std::copy(data, data + element_count, host_data);
}

bool MnnRuntimeClass::CanSetInputFromFloat(int index) const {
    if (index < 0 || index >= GetInputCount()) {
        return false;
    }
    return input_infos_[static_cast<std::size_t>(index)].element_type == TensorElementType::FLOAT32;
}

void MnnRuntimeClass::SetInput(int index, const TensorView &input) {
    if (index < 0 || index >= GetInputCount()) {
        throw std::out_of_range("input index out of range");
    }
    if (!input.data) {
        throw std::invalid_argument("input data is null");
    }

    const auto &info = input_infos_[static_cast<std::size_t>(index)];
    if (input.element_type != info.element_type) {
        throw std::invalid_argument("input tensor dtype mismatch");
    }
    if (input.element_count != static_cast<std::size_t>(info.total_size)) {
        throw std::invalid_argument("input element count mismatch");
    }
    const auto expected_bytes = TensorByteCount(info.element_type, input.element_count);
    if (input.byte_count != expected_bytes) {
        throw std::invalid_argument("input tensor byte count mismatch");
    }

    void *host_data = imp_->host_input_tensors[static_cast<std::size_t>(index)]->host<void>();
    std::memcpy(host_data, input.data, expected_bytes);
}

const std::vector<float> &MnnRuntimeClass::GetOutput(int index) const {
    EnsureOutputsValid();

    if (index < 0 || index >= GetOutputCount()) {
        throw std::out_of_range("output index out of range");
    }

    const std::size_t idx = static_cast<std::size_t>(index);
    const auto &info = output_infos_[idx];
    if (info.element_type != TensorElementType::FLOAT32) {
        throw std::runtime_error(
            "output tensor " + info.name + " cannot be converted to float");
    }
    auto &view = output_float_views_[idx];

    const float *output_data = imp_->host_output_tensors[idx]->host<float>();
    view.assign(output_data, output_data + info.total_size);

    return view;
}

bool MnnRuntimeClass::CanGetOutputAsFloat(int index) const {
    if (index < 0 || index >= GetOutputCount()) {
        return false;
    }
    return output_infos_[static_cast<std::size_t>(index)].element_type == TensorElementType::FLOAT32;
}

TensorView MnnRuntimeClass::GetOutputView(int index) const {
    EnsureOutputsValid();

    if (index < 0 || index >= GetOutputCount()) {
        throw std::out_of_range("output index out of range");
    }

    const std::size_t idx = static_cast<std::size_t>(index);
    const auto &info = output_infos_[idx];
    const void *output_data = imp_->host_output_tensors[idx]->host<void>();

    return {
        info.element_type,
        output_data,
        static_cast<std::size_t>(info.total_size),
        TensorByteCount(info.element_type, static_cast<std::size_t>(info.total_size)),
    };
}

void MnnRuntimeClass::CopyOutputToInput(int output_index, int input_index) {
    EnsureOutputsValid();

    if (output_index < 0 || output_index >= GetOutputCount()) {
        throw std::out_of_range("output index out of range");
    }
    if (input_index < 0 || input_index >= GetInputCount()) {
        throw std::out_of_range("input index out of range");
    }

    const auto &out_info = output_infos_[static_cast<std::size_t>(output_index)];
    const auto &in_info = input_infos_[static_cast<std::size_t>(input_index)];

    if (out_info.element_type != in_info.element_type ||
        out_info.total_size != in_info.total_size) {
        throw std::invalid_argument("output-input dtype or size mismatch for feedback");
    }

    const auto byte_count = TensorByteCount(
        out_info.element_type, static_cast<std::size_t>(out_info.total_size));
    const void *output_data =
        imp_->host_output_tensors[static_cast<std::size_t>(output_index)]->host<void>();
    void *input_data =
        imp_->host_input_tensors[static_cast<std::size_t>(input_index)]->host<void>();
    std::memcpy(input_data, output_data, byte_count);
}

void MnnRuntimeClass::PrintModelInfo() const {
    std::cout << "MNN Model Information:\n";
    std::cout << "  Backend: " << runtime_info_.initialized_backend << "\n";
    std::cout << "  Threads: " << runtime_info_.threads << "\n";
    std::cout << "  High Precision: " << (runtime_info_.high_precision ? "yes" : "no") << "\n\n";

    std::cout << "Inputs (" << input_infos_.size() << "):\n";
    for (const auto &info : input_infos_) {
        std::cout << "  - " << info.name << " [";
        for (size_t i = 0; i < info.shape.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << info.shape[i];
        }
        std::cout << "] " << info.element_type_name << " (total: " << info.total_size << ")\n";
    }

    std::cout << "\nOutputs (" << output_infos_.size() << "):\n";
    for (const auto &info : output_infos_) {
        std::cout << "  - " << info.name << " [";
        for (size_t i = 0; i < info.shape.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << info.shape[i];
        }
        std::cout << "] " << info.element_type_name << " (total: " << info.total_size << ")\n";
    }
}

void MnnRuntimeClass::EnsureOutputsValid() const {
    if (!outputs_valid_) {
        throw std::runtime_error("outputs not valid: Run() not called or failed");
    }
}

}  // namespace mnn_runtime
