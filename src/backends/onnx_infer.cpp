/**
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file onnx_infer.cpp
 * @brief ONNX Runtime 推理实现
 */

#include "onnx_infer.h"

#include <onnxruntime_cxx_api.h>

#include <cstring>
#include <iostream>
#include <numeric>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if defined(cpu_rv64) || defined(__riscv)
#include "spacemit_ort_env.h"
#endif
namespace onnx_runtime {

class OnnxRuntimeClass::ImpClass {
public:
    explicit ImpClass(OnnxRuntimeClass *omp);
    ~ImpClass();
    bool Init(const std::string &model_file);
    void Step();

    OnnxRuntimeClass &omp;

    // ONNX Runtime相关成员
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
};

// ImpClass构造函数
OnnxRuntimeClass::ImpClass::ImpClass(OnnxRuntimeClass *omp)
    : omp(*omp), env(ORT_LOGGING_LEVEL_WARNING, "OnnxRuntime") {
    std::cout << "[ONNX Runtime] ImpClass 构造" << std::endl;
}

OnnxRuntimeClass::ImpClass::~ImpClass() {
    std::cout << "[ONNX Runtime] ImpClass 析构" << std::endl;
}

// 解析动态维度（负值→1），返回解析后的 shape
static std::vector<int64_t> ResolveShape(const std::vector<int64_t> &shape) {
    std::vector<int64_t> resolved = shape;
    for (auto &dim : resolved) {
        if (dim <= 0)
            dim = 1;  // 动态维度默认为1
    }
    return resolved;
}

// 计算张量总元素数
static int64_t ComputeTensorSize(const std::vector<int64_t> &shape) {
    if (shape.empty())
        return 0;
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

// ImpClass::Init函数 - 自动推断版本
bool OnnxRuntimeClass::ImpClass::Init(const std::string &model_file) {
    std::cout << "[ONNX Runtime] 开始初始化模型: " << model_file << std::endl;

    try {
        // 设置会话选项
#if defined(cpu_rv64) || defined(__riscv)
        SessionOptionsSpaceMITEnvInit(session_options);  // 可选：SpaceMIT 专属 EP
#endif

        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.SetIntraOpNumThreads(1);
        session_options.SetInterOpNumThreads(1);

        // 加载模型
        session = std::make_unique<Ort::Session>(env, model_file.c_str(), session_options);
        std::cout << "[ONNX Runtime] 模型加载成功" << std::endl;

        // ============================================================
        // 自动推断输入信息
        // ============================================================
        size_t input_count = session->GetInputCount();
        std::cout << "[ONNX Runtime] 检测到 " << input_count << " 个输入" << std::endl;

        omp.input_infos_.resize(input_count);
        omp.inputs_.resize(input_count);

        for (size_t i = 0; i < input_count; ++i) {
            // 获取输入名称
            auto input_name = session->GetInputNameAllocated(i, allocator);
            omp.input_infos_[i].name = input_name.get();

            // 获取输入维度并解析动态维度
            auto type_info = session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            omp.input_infos_[i].shape = ResolveShape(tensor_info.GetShape());

            // 计算总元素数
            omp.input_infos_[i].total_size = ComputeTensorSize(omp.input_infos_[i].shape);

            // 初始化输入向量
            omp.inputs_[i].setZero(omp.input_infos_[i].total_size);

            // 建立名称到索引的映射
            omp.input_name_to_index_[omp.input_infos_[i].name] = i;

            std::cout << "  输入[" << i << "]: " << omp.input_infos_[i].name << ", shape=[";
            for (size_t j = 0; j < omp.input_infos_[i].shape.size(); ++j) {
                std::cout << omp.input_infos_[i].shape[j];
                if (j < omp.input_infos_[i].shape.size() - 1)
                    std::cout << ", ";
            }
            std::cout << "], total_size=" << omp.input_infos_[i].total_size << std::endl;
        }

        // ============================================================
        // 自动推断输出信息
        // ============================================================
        size_t output_count = session->GetOutputCount();
        std::cout << "[ONNX Runtime] 检测到 " << output_count << " 个输出" << std::endl;

        omp.output_infos_.resize(output_count);
        omp.outputs_.resize(output_count);

        for (size_t i = 0; i < output_count; ++i) {
            // 获取输出名称
            auto output_name = session->GetOutputNameAllocated(i, allocator);
            omp.output_infos_[i].name = output_name.get();

            // 获取输出维度并解析动态维度
            auto type_info = session->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            omp.output_infos_[i].shape = ResolveShape(tensor_info.GetShape());

            // 计算总元素数
            omp.output_infos_[i].total_size = ComputeTensorSize(omp.output_infos_[i].shape);

            // 初始化输出向量
            omp.outputs_[i].setZero(omp.output_infos_[i].total_size);

            // 建立名称到索引的映射
            omp.output_name_to_index_[omp.output_infos_[i].name] = i;

            std::cout << "  输出[" << i << "]: " << omp.output_infos_[i].name << ", shape=[";
            for (size_t j = 0; j < omp.output_infos_[i].shape.size(); ++j) {
                std::cout << omp.output_infos_[i].shape[j];
                if (j < omp.output_infos_[i].shape.size() - 1)
                    std::cout << ", ";
            }
            std::cout << "], total_size=" << omp.output_infos_[i].total_size << std::endl;
        }

        std::cout << "[ONNX Runtime] 初始化完成" << std::endl;
        return true;
    } catch (const Ort::Exception &e) {
        std::cerr << "[ONNX Runtime] 初始化错误: " << e.what() << std::endl;
        return false;
    } catch (const std::exception &e) {
        std::cerr << "[ONNX Runtime] 初始化异常: " << e.what() << std::endl;
        return false;
    }
}

// ImpClass::Step函数 - 通用推理
void OnnxRuntimeClass::ImpClass::Step() {
    if (!session) {
        std::cerr << "[ONNX Runtime] 错误: 会话未初始化" << std::endl;
        return;
    }

    try {
        // 准备输入张量
        std::vector<Ort::Value> input_tensors;
        std::vector<const char *> input_names;
        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        for (size_t i = 0; i < omp.inputs_.size(); ++i) {
            input_names.push_back(omp.input_infos_[i].name.c_str());

            Ort::Value tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                                omp.inputs_[i].data(),
                                                                omp.input_infos_[i].total_size,
                                                                omp.input_infos_[i].shape.data(),
                                                                omp.input_infos_[i].shape.size());
            input_tensors.push_back(std::move(tensor));
        }

        // 准备输出名称
        std::vector<const char *> output_names;
        for (size_t i = 0; i < omp.outputs_.size(); ++i) {
            output_names.push_back(omp.output_infos_[i].name.c_str());
        }

        // 执行推理
        Ort::RunOptions run_options;
        auto output_tensors = session->Run(run_options,
                                            input_names.data(),
                                            input_tensors.data(),
                                            input_tensors.size(),
                                            output_names.data(),
                                            output_names.size());

        // 拷贝输出数据
        for (size_t i = 0; i < output_tensors.size(); ++i) {
            if (output_tensors[i].IsTensor()) {
                float *output_data = output_tensors[i].GetTensorMutableData<float>();
                std::memcpy(omp.outputs_[i].data(),
                            output_data,
                            omp.output_infos_[i].total_size * sizeof(float));
            } else {
                std::cerr << "[ONNX Runtime] 警告: 输出[" << i << "] 不是张量" << std::endl;
                omp.outputs_[i].setZero();
            }
        }
    } catch (const Ort::Exception &e) {
        std::cerr << "[ONNX Runtime] 推理错误: " << e.what() << std::endl;
        for (auto &output : omp.outputs_) {
            output.setZero();
        }
    } catch (const std::exception &e) {
        std::cerr << "[ONNX Runtime] 推理异常: " << e.what() << std::endl;
        for (auto &output : omp.outputs_) {
            output.setZero();
        }
    }
}

// OnnxRuntimeClass 公共接口实现
OnnxRuntimeClass::OnnxRuntimeClass() : imp_(std::make_unique<ImpClass>(this)) {
    std::cout << "[ONNX Runtime] OnnxRuntimeClass 构造" << std::endl;
}

OnnxRuntimeClass::~OnnxRuntimeClass() {
    std::cout << "[ONNX Runtime] OnnxRuntimeClass 析构" << std::endl;
}

bool OnnxRuntimeClass::Init(const std::string &model_file) {
    if (!imp_) {
        std::cerr << "[ONNX Runtime] 错误: ImpClass 未初始化" << std::endl;
        return false;
    }
    return imp_->Init(model_file);
}

void OnnxRuntimeClass::Run() {
    if (!imp_) {
        std::cerr << "[ONNX Runtime] 错误: ImpClass 未初始化" << std::endl;
        return;
    }
    imp_->Step();
}

// 获取模型信息
int OnnxRuntimeClass::GetInputCount() const {
    return static_cast<int>(inputs_.size());
}

int OnnxRuntimeClass::GetOutputCount() const {
    return static_cast<int>(outputs_.size());
}

const TensorInfo &OnnxRuntimeClass::GetInputInfo(int index) const {
    return input_infos_.at(index);
}

const TensorInfo &OnnxRuntimeClass::GetOutputInfo(int index) const {
    return output_infos_.at(index);
}

// 通过索引访问
vecXf &OnnxRuntimeClass::GetInput(int index) {
    return inputs_.at(index);
}

vecXf &OnnxRuntimeClass::GetOutput(int index) {
    return outputs_.at(index);
}

// 通过名称访问
vecXf &OnnxRuntimeClass::GetInput(const std::string &name) {
    auto it = input_name_to_index_.find(name);
    if (it == input_name_to_index_.end()) {
        throw std::runtime_error("[ONNX Runtime] 输入名称不存在: " + name);
    }
    return inputs_[it->second];
}

vecXf &OnnxRuntimeClass::GetOutput(const std::string &name) {
    auto it = output_name_to_index_.find(name);
    if (it == output_name_to_index_.end()) {
        throw std::runtime_error("[ONNX Runtime] 输出名称不存在: " + name);
    }
    return outputs_[it->second];
}

// 打印模型信息
void OnnxRuntimeClass::PrintModelInfo() const {
    std::cout << "\n========== ONNX 模型信息 ==========" << std::endl;
    std::cout << "输入数量: " << GetInputCount() << std::endl;
    for (int i = 0; i < GetInputCount(); ++i) {
        const auto &info = input_infos_[i];
        std::cout << "  [" << i << "] " << info.name << ": [";
        for (size_t j = 0; j < info.shape.size(); ++j) {
            std::cout << info.shape[j];
            if (j < info.shape.size() - 1)
                std::cout << ", ";
        }
        std::cout << "] (total: " << info.total_size << ")" << std::endl;
    }

    std::cout << "输出数量: " << GetOutputCount() << std::endl;
    for (int i = 0; i < GetOutputCount(); ++i) {
        const auto &info = output_infos_[i];
        std::cout << "  [" << i << "] " << info.name << ": [";
        for (size_t j = 0; j < info.shape.size(); ++j) {
            std::cout << info.shape[j];
            if (j < info.shape.size() - 1)
                std::cout << ", ";
        }
        std::cout << "] (total: " << info.total_size << ")" << std::endl;
    }
    std::cout << "===================================" << std::endl;
}

}  // namespace onnx_runtime
