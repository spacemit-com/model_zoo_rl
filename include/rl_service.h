/**
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file rl_service.h
 * @brief RL 策略执行器公共接口
 *
 * rl_service 是 rl 模块的对外统一接口：
 *   - 模型加载（内部委托推理后端，当前支持 ONNX Runtime）
 *   - 段式观测组装（通过可插拔的 ObsSegmentAssembler 策略）
 *   - 声明式模型 I/O 绑定与推理执行
 *   - 动作映射
 *
 * 观测组装基于「段拼接」模型：
 *   每个段（segment）独立指定一组 obs terms 和一种基础组装模式：
 *     - (默认)        无历史，输出当前帧
 *     - flat_history   按变量分组的环形历史缓冲
 *     - frame_history  按帧的滑窗历史缓冲
 *   多段按顺序拼接为完整观测向量。
 */
#ifndef RL_SERVICE_H
#define RL_SERVICE_H

#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace rl_policy {

/** @brief 模型张量元素类型 */
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

/** @brief 非持有的原生模型张量视图 */
struct TensorView {
    TensorElementType element_type = TensorElementType::UNDEFINED;
    const void *data = nullptr;
    std::size_t element_count = 0;
    std::size_t byte_count = 0;
};

template <typename T>
struct TensorElementTypeOf;

#define RL_DEFINE_TENSOR_TYPE(cpp_type, tensor_type) \
    template <>                                       \
    struct TensorElementTypeOf<cpp_type> {            \
        static constexpr TensorElementType value = TensorElementType::tensor_type; \
    }

RL_DEFINE_TENSOR_TYPE(float, FLOAT32);
RL_DEFINE_TENSOR_TYPE(double, FLOAT64);
RL_DEFINE_TENSOR_TYPE(std::uint8_t, UINT8);
RL_DEFINE_TENSOR_TYPE(std::int8_t, INT8);
RL_DEFINE_TENSOR_TYPE(std::uint16_t, UINT16);
RL_DEFINE_TENSOR_TYPE(std::int16_t, INT16);
RL_DEFINE_TENSOR_TYPE(std::uint32_t, UINT32);
RL_DEFINE_TENSOR_TYPE(std::int32_t, INT32);
RL_DEFINE_TENSOR_TYPE(std::uint64_t, UINT64);
RL_DEFINE_TENSOR_TYPE(std::int64_t, INT64);
RL_DEFINE_TENSOR_TYPE(bool, BOOL);

#undef RL_DEFINE_TENSOR_TYPE

/** @brief 为普通 C++ 标量数组构造原生张量视图 */
template <typename T>
TensorView MakeTensorView(const T *data, std::size_t element_count) {
    return {
        TensorElementTypeOf<T>::value,
        data,
        element_count,
        element_count * sizeof(T),
    };
}

// ============================================================
// 观测段配置
// ============================================================

/**
 * @brief 单个观测段的配置
 *
 * mode:
 *   - ""  / 未设置 : 无历史，仅输出当前帧
 *   - "flat_history" : 按变量分组的环形历史
 *   - "frame_history": 按帧的滑窗历史
 *
 * order:
 *   - "oldest_first" : [t-N, ..., t-1]
 *   - "newest_first" : [t-1, ..., t-N]
 *
 * include_current:
 *   - true  : 先写入当前帧再读取历史（RoboMimic 风格）
 *   - false : 先读取历史再写入当前帧（青龙风格）
 */
struct ObsSegmentConfig {
    std::vector<std::string> terms;      ///< 观测项列表
    std::string mode;                    ///< "" | "flat_history" | "frame_history"
    int length = 0;                      ///< 历史帧数（flat_history / frame_history 使用）
    std::string order = "oldest_first";  ///< 历史排序方向
    bool include_current = true;         ///< 是否在读取前先写入当前帧
};

/** @brief ONNX 输入张量的数据来源 */
enum class ModelInputSource {
    OBSERVATION,          ///< 从 AssembleObs 生成的观测向量取一段
    OBSERVATION_HISTORY,  ///< 自动维护指定输入张量的历史窗口
    FEEDBACK,             ///< 上一帧输出反馈到下一帧输入
    CONSTANT,             ///< 固定值
    EXTERNAL,             ///< 上层通过 SetModelInput 注入
};

/** @brief ONNX 输出张量的处理方式 */
enum class ModelOutputTarget {
    ACTION,  ///< 策略动作
    EXPOSE,  ///< 通过 GetModelOutput 暴露给上层
    IGNORE,  ///< 明确忽略
};

/** @brief 单个 ONNX 输入张量的声明式绑定 */
struct ModelInputBindingConfig {
    std::string name;  ///< ONNX 输入张量名
    ModelInputSource source = ModelInputSource::EXTERNAL;
    std::string key;              ///< EXTERNAL 输入的上层键名，空则使用 name
    std::string feedback_output;  ///< FEEDBACK 对应的 ONNX 输出张量名
    std::string history_source;   ///< HISTORY 跟踪的 OBSERVATION 输入张量名
    int observation_offset = -1;  ///< OBSERVATION 在完整观测中的起始位置，-1 自动顺排
    std::vector<float> initial_value;  ///< CONSTANT 值或 EXTERNAL/FEEDBACK 初始值
};

/** @brief 单个 ONNX 输出张量的声明式绑定 */
struct ModelOutputBindingConfig {
    std::string name;  ///< ONNX 输出张量名
    ModelOutputTarget target = ModelOutputTarget::EXPOSE;
    std::string key;  ///< EXPOSE 的上层键名，空则使用 name
};

/**
 * @brief 模型 I/O 拓扑配置
 *
 * 每个 ONNX 输入必须有唯一来源，每个输出必须声明用途或被 feedback 引用；
 * 输出可同时作为 action/expose 和 feedback 源。
 */
struct ModelIOConfig {
    std::vector<ModelInputBindingConfig> inputs;
    std::vector<ModelOutputBindingConfig> outputs;
};

/** @brief ONNX Runtime provider 与线程配置。 */
struct InferenceRuntimeConfig {
    std::string provider = "auto";  ///< auto | cpu | spacemit
    int threads = 1;                ///< CPU intra-op 或 SpaceMIT EP 线程数
    std::string affinity;           ///< SpaceMIT EP CPU 列表，分号分隔，例如 "0;1"
    bool ep_dump_subgraphs = false;  ///< 导出 SpaceMIT EP 实际编译子图
    std::string ep_profile_prefix;   ///< 非空时导出 SpaceMIT EP profile JSON
    bool ort_spinning = true;        ///< ONNX Runtime intra-op worker 是否允许 busy-spin
};

/** @brief 实际初始化出的推理会话信息。 */
struct InferenceRuntimeInfo {
    std::string requested_provider;
    std::string initialized_provider;
    int ort_intra_threads = 1;
    int ort_inter_threads = 1;
    int ep_threads = 0;             ///< 传给 EP 的请求值，不代表实际并行核数
    std::string affinity;           ///< 传给 EP 的请求值，不代表实际线程落核
    std::string provider_status;
    bool ort_spinning = true;
};

// ============================================================
// 策略执行器配置（YAML 驱动）
// ============================================================

struct PolicyExecutorConfig {
    // ---- 模型与动作 ----
    std::string model_path;
    std::vector<double> action_scale = {0.25};
    double action_blend_ratio = 1.0;
    std::vector<double> rl_default_pos;
    std::vector<int> action_joint_index;

    // ---- 推理 backend ----
    InferenceRuntimeConfig runtime;

    // ---- 模型 I/O 拓扑 ----
    ModelIOConfig model_io;

    // ---- 可选的数值裁剪（空=禁用，单值=全维，或逐维配置）----
    std::vector<double> clip_observations;
    std::vector<double> clip_actions;

    // ---- 段式观测配置（必须） ----
    std::vector<ObsSegmentConfig> obs_segments;

    // ---- 观测归一化参数 ----
    double ang_vel_scale = 1.0;
    double dof_pos_scale = 1.0;
    double dof_vel_scale = 1.0;
    double euler_angle_scale = 1.0;
    std::array<double, 3> command_scale = {1.0, 1.0, 1.0};
    bool dof_pos_subtract_default = true;

    // ---- phase / gait_phase 参数 ----
    double phase_period = 1.0;
    double gait_cycle = 0.85;
    double gait_left_offset = 0.0;
    double gait_right_offset = 0.5;
    double gait_left_ratio = 0.5;
    double gait_right_ratio = 0.5;

    // ---- ref_motion_phase 参数 ----
    double motion_length = 0.0;

    // ---- 自定义标量默认值 ----
    std::unordered_map<std::string, float> custom_scalar_defaults;

    // ---- 自定义数组维度声明（泛型扩展点，N 维 obs term 通过 SetCustomArray 注入）----
    // 例：tracking 策略声明 {"motion_command":58,"motion_anchor_pos_b":3,"motion_anchor_ori_b":6}
    std::unordered_map<std::string, int> custom_array_dims;

    // ---- 维度校验 ----
    bool strict_obs_dim_check = false;
};

// ============================================================
// 策略 YAML 解析结果
// ============================================================

struct LoadedPolicyConfig {
    PolicyExecutorConfig exec_cfg;
    std::array<double, 3> command_init = {0.0, 0.0, 0.0};
    double rl_dt = 0.02;
    int infer_decimation = 4;
    double max_roll = 0.7;
    double max_pitch = 0.7;
    std::vector<double> kp;  // 策略训练时对应的 PD 刚度，为空表示未配置
    std::vector<double> kd;  // 策略训练时对应的 PD 阻尼，为空表示未配置
};

/**
 * @brief 从 YAML 加载指定策略配置
 *
 * 会解析并校验：
 * - 模型路径、动作映射、观测段配置
 * - command.scale / command.init
 * - infer_decimation / max_roll / max_pitch / thread
 *
 * @param yaml_path   YAML 配置文件路径
 * @param policy_name 策略名（对应 rl_policy.onnx_infer.<policy_name>）
 * @param robot_dir   机器人资源根目录（绝对路径），用于解析 model_path 等相对路径
 */
LoadedPolicyConfig LoadPolicyConfigFromYaml(const std::string &yaml_path,
                                            const std::string &policy_name,
                                            const std::string &robot_dir);

// ============================================================
// 策略执行器
// ============================================================

class PolicyExecutor {
public:
    PolicyExecutor();
    ~PolicyExecutor();

    // 禁止拷贝，允许移动
    PolicyExecutor(const PolicyExecutor &) = delete;
    PolicyExecutor &operator=(const PolicyExecutor &) = delete;
    PolicyExecutor(PolicyExecutor &&) noexcept;
    PolicyExecutor &operator=(PolicyExecutor &&) noexcept;

    /**
     * @brief 初始化策略执行器
     * @param cfg 策略配置（模型路径、动作映射、观测段等）
     */
    void Init(const PolicyExecutorConfig &cfg);

    /** @return 观测向量维度 */
    int ObsDim() const;

    /** @return 动作向量维度 */
    int ActionDim() const;

    /** @return 自动反馈状态张量对数量 */
    int FeedbackStateCount() const;

    /** @return 是否使用 observation_history 输入 */
    bool HasObsHist() const;

    /** @return 实际初始化出的推理 provider 与线程信息。 */
    InferenceRuntimeInfo GetRuntimeInfo() const;

    /** @brief 打印模型信息 */
    void PrintModelInfo() const;

    /** 设置自定义标量（如 "z", "stand_flag"），需在 AssembleObs 前调用 */
    void SetCustomScalar(const std::string &name, float value);
    float GetCustomScalar(const std::string &name) const;

    /**
     * @brief 推入自定义 N 维数组 obs term（泛型扩展点）
     *
     * 用于上层（如 tracking 状态）每帧把外部计算好的多维数据 push 进 obs。
     * 维度需在 PolicyExecutorConfig.custom_array_dims 中先声明（Init 时已注册）。
     *
     * 例：tracking 策略每帧调用：
     *   policy.SetCustomArray("motion_command", motion_buf_58, 58);
     *   policy.SetCustomArray("motion_anchor_pos_b", anchor_pos_3, 3);
     *   policy.SetCustomArray("motion_anchor_ori_b", anchor_ori_6, 6);
     *
     * 需在 AssembleObs 前调用。
     */
    void SetCustomArray(const std::string &name, const float *data, int size);

    /**
     * @brief 设置 model_io 中 source=external 的输入张量
     * @param key 绑定配置的 key；未配置 key 时使用 ONNX 张量名
     */
    void SetModelInput(const std::string &key, const float *data, int size);

    /**
     * @brief 使用与模型声明一致的原生 dtype 设置 external 输入
     * @param key 绑定配置的 key；未配置 key 时使用 ONNX 张量名
     * @param input 非持有 typed view，数据在调用期间保持有效即可
     */
    void SetModelInput(const std::string &key, const TensorView &input);

    /**
     * @brief 读取 model_io 中 target=expose 的最近一次输出
     * @throws std::runtime_error key 不存在或尚未完成推理
     */
    const std::vector<float> &GetModelOutput(const std::string &key) const;

    /**
     * @brief 读取 expose 输出的原生 typed view
     * @note view 指向内部缓冲区，有效期至下一次 Infer
     */
    TensorView GetModelOutputTensor(const std::string &key) const;

    /**
     * @brief 组装观测向量
     *
     * 按 segments 配置顺序：计算每段当前帧 → 交给对应 assembler → 拼接输出
     */
    void AssembleObs(const std::array<double, 3> &gyro,
                    const std::array<double, 3> &rpy,
                    double cmd_vx,
                    double cmd_vy,
                    double cmd_wz,
                    const std::vector<double> &joint_pos,
                    const std::vector<double> &joint_vel,
                    const std::array<double, 4> &base_quat,
                    const std::array<double, 3> &base_vel,
                    float dt,
                    Eigen::VectorXf &out_obs);

    /** 按 model_io 绑定执行推理 */
    void Infer(const Eigen::VectorXf &obs, std::vector<double> &out_action);

    /**
     * @brief 执行推理，并可同时返回动作裁剪和平滑前的 ONNX 原始输出
     * @param obs 组装后的观测向量
     * @param out_action 经过 clip_actions 和 action_blend_ratio 后的动作
     * @param raw_action ONNX action 输出；必须与 out_action 使用不同容器
    */
    void Infer(const Eigen::VectorXf &obs, std::vector<double> &out_action,
                std::vector<double> *raw_action);

    /** 将策略动作映射为全身关节目标位置 */
    void MapActionToTargetPos(const std::vector<double> &action,
                            std::vector<double> &target_pos) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rl_policy

#endif  // RL_SERVICE_H
