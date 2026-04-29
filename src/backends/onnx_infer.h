/**
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file onnx_infer.h
 * @brief ONNX Runtime 推理封装
 *
 * 提供模型加载、自动推断输入输出维度、推理执行等功能。
 * 内部使用 Pimpl 模式隔离 ONNX Runtime 依赖。
 */
#ifndef ONNX_INFER_H
#define ONNX_INFER_H

#include <Eigen/Dense>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace onnx_runtime {

using vecXf = Eigen::VectorXf;

/**
 * @brief 张量信息（输入/输出的名称、形状、元素总数）
 */
struct TensorInfo {
    std::string name;            ///< 张量名称
    std::vector<int64_t> shape;  ///< 维度形状
    int64_t total_size;          ///< 所有元素总数
};

/**
 * @brief ONNX Runtime 推理封装类
 *
 * 支持自动推断模型输入输出维度，通过索引或名称访问输入输出张量。
 */
class OnnxRuntimeClass {
public:
    OnnxRuntimeClass();
    ~OnnxRuntimeClass();

    /**
     * @brief 初始化模型（自动推断输入输出信息）
     * @param model_file ONNX 模型文件路径
     * @return 成功返回 true
     */
    bool Init(const std::string &model_file);

    /** @brief 执行一次推理 */
    void Run();

    /** @return 模型输入个数 */
    int GetInputCount() const;

    /** @return 模型输出个数 */
    int GetOutputCount() const;

    /**
     * @brief 获取输入张量信息
     * @param index 输入索引
     * @return 张量信息引用
     */
    const TensorInfo &GetInputInfo(int index) const;

    /**
     * @brief 获取输出张量信息
     * @param index 输出索引
     * @return 张量信息引用
     */
    const TensorInfo &GetOutputInfo(int index) const;

    /**
     * @brief 通过索引访问输入张量（推荐）
     * @param index 输入索引
     * @return 输入向量引用（可修改）
     */
    vecXf &GetInput(int index);

    /**
     * @brief 通过索引访问输出张量
     * @param index 输出索引
     * @return 输出向量引用
     */
    vecXf &GetOutput(int index);

    /**
     * @brief 通过名称访问输入张量
     * @param name 输入名称
     * @return 输入向量引用
     * @throws std::runtime_error 名称不存在时抛出异常
     */
    vecXf &GetInput(const std::string &name);

    /**
     * @brief 通过名称访问输出张量
     * @param name 输出名称
     * @return 输出向量引用
     * @throws std::runtime_error 名称不存在时抛出异常
     */
    vecXf &GetOutput(const std::string &name);

    /** @brief 打印模型输入输出信息 */
    void PrintModelInfo() const;

private:
    class ImpClass;
    std::unique_ptr<ImpClass> imp_;

    // 输入输出数据容器
    std::vector<vecXf> inputs_;
    std::vector<vecXf> outputs_;

    // 输入输出信息
    std::vector<TensorInfo> input_infos_;
    std::vector<TensorInfo> output_infos_;

    // 名称到索引的映射
    std::map<std::string, int> input_name_to_index_;
    std::map<std::string, int> output_name_to_index_;
};

}  // namespace onnx_runtime

#endif  // ONNX_INFER_H
