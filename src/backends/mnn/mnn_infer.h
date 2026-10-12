/**
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file mnn_infer.h
 * @brief MNN 推理封装
 *
 * 提供模型加载、自动推断输入输出维度、推理执行等功能。
 * 内部使用 Pimpl 模式隔离 MNN 依赖。
 */
#ifndef MNN_INFER_H
#define MNN_INFER_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mnn_runtime {

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

/** @brief MNN Runtime 会话配置。 */
struct RuntimeOptions {
    std::string backend = "cpu";  ///< cpu | auto
    int threads = 1;              ///< CPU 线程数
    bool high_precision = true;   ///< 使用高精度模式
};

/** @brief 实际创建出的 MNN Runtime 会话信息。 */
struct RuntimeInfo {
    std::string requested_backend = "cpu";
    std::string initialized_backend = "cpu";
    int threads = 1;
    bool high_precision = true;
    std::string backend_status;
};

/**
 * @brief MNN Runtime 推理封装类
 *
 * 支持自动推断模型输入输出维度，并通过索引访问输入输出张量。
 */
class MnnRuntimeClass {
public:
    MnnRuntimeClass();
    ~MnnRuntimeClass();

    /**
     * @brief 初始化模型（自动推断输入输出信息）
     * @param model_file MNN 模型文件路径
     * @return 成功返回 true
     */
    bool Init(const std::string &model_file);

    /**
     * @brief 使用显式 backend、线程数等配置初始化模型。
     * @return 成功返回 true
     */
    bool Init(const std::string &model_file, const RuntimeOptions &options);

    /**
     * @brief 执行一次推理
     * @return 成功返回 true，失败返回 false，详情通过 GetLastError() 获取
     */
    bool Run();

    /** @brief 请求中断当前正在执行的推理。再次使用前需重新 Init。 */
    void RequestTerminate();

    /** @return 最近一次初始化或推理失败的错误信息 */
    const std::string &GetLastError() const;

    /** @return 当前会话的 backend 与线程配置。 */
    const RuntimeInfo &GetRuntimeInfo() const;

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

    /** 设置 FLOAT32 输入；其他 dtype 请使用原生 TensorView。 */
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
    RuntimeInfo runtime_info_;
    std::atomic_bool terminate_requested_{false};
};

}  // namespace mnn_runtime

#endif  // MNN_INFER_H
