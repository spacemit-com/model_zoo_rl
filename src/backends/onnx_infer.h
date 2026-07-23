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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace onnx_runtime {

/** @brief Backend 内部张量元素类型 */
enum class TensorElementType : int {
    UNDEFINED = 0,
    FLOAT32 = 1,
    UINT8 = 2,
    INT8 = 3,
    UINT16 = 4,
    INT16 = 5,
    INT32 = 6,
    INT64 = 7,
    STRING = 8,
    BOOL = 9,
    FLOAT16 = 10,
    FLOAT64 = 11,
    UINT32 = 12,
    UINT64 = 13,
    COMPLEX64 = 14,
    COMPLEX128 = 15,
    BFLOAT16 = 16,
    FLOAT8E4M3FN = 17,
    FLOAT8E4M3FNUZ = 18,
    FLOAT8E5M2 = 19,
    FLOAT8E5M2FNUZ = 20,
    UINT4 = 21,
    INT4 = 22,
};

/** @brief Backend 内部非持有张量视图 */
struct TensorView {
    TensorElementType element_type = TensorElementType::UNDEFINED;
    const void *data = nullptr;
    std::size_t element_count = 0;
    std::size_t byte_count = 0;
};

/**
 * @brief 张量信息（输入/输出的名称、形状、元素总数）
 */
struct TensorInfo {
    std::string name;                    ///< 张量名称
    std::vector<int64_t> shape;          ///< 维度形状
    int64_t total_size = 0;              ///< 所有元素总数
    TensorElementType element_type = TensorElementType::UNDEFINED;  ///< 张量元素类型
    std::string element_type_name;       ///< 便于日志展示的 dtype 名称
};

/**
 * @brief ONNX Runtime 推理封装类
 *
 * 支持自动推断模型输入输出维度，并通过索引访问输入输出张量。
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

    /**
     * @brief 执行一次推理
     * @return 成功返回 true，失败返回 false，详情通过 GetLastError() 获取
     */
    bool Run();

    /** @return 最近一次初始化或推理失败的错误信息 */
    const std::string &GetLastError() const;

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

    /** 使用 float 语义值设置输入，按模型声明的 dtype 转换。 */
    void SetInputFromFloat(int index, const float *data, std::size_t element_count);

    /** @return 指定输入是否支持从 float 语义值转换。 */
    bool CanSetInputFromFloat(int index) const;

    /** 使用与模型 dtype 完全一致的原生数据设置输入。 */
    void SetInput(int index, const TensorView &input);

    /** 将最近一次成功推理的输出转换为 float 便捷视图。 */
    const std::vector<float> &GetOutput(int index) const;

    /** @return 指定输出是否支持转换为 float 便捷视图。 */
    bool CanGetOutputAsFloat(int index) const;

    /** 获取最近一次成功推理的原生 typed view，有效期至下次 Run。 */
    TensorView GetOutputView(int index) const;

    /** 将有效输出原样复制到下一帧输入，用于 recurrent feedback。 */
    void CopyOutputToInput(int output_index, int input_index);

    /** @brief 打印模型输入输出信息 */
    void PrintModelInfo() const;

private:
    void EnsureOutputsValid() const;

    class ImpClass;
    std::unique_ptr<ImpClass> imp_;

    mutable std::vector<std::vector<float>> output_float_views_;

    // 输入输出信息
    std::vector<TensorInfo> input_infos_;
    std::vector<TensorInfo> output_infos_;

    std::string last_error_;
    bool outputs_valid_ = false;
};

}  // namespace onnx_runtime

#endif  // ONNX_INFER_H
