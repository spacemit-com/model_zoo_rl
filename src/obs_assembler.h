/**
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file obs_assembler.h
 * @brief 观测段组装器抽象基类
 *
 * 每个 ObsSegmentAssembler 负责一个段的历史管理与输出。
 * 三种具体实现：
 *   - ObsNone         : 无历史，直接输出当前帧
 *   - ObsFlatHistory   : 按变量分组的环形历史缓冲
 *   - ObsFrameHistory  : 按帧的滑窗历史缓冲
 */

#ifndef OBS_ASSEMBLER_H
#define OBS_ASSEMBLER_H


#include <memory>
#include <string>
#include <vector>

namespace rl_policy {

/// 描述一个 obs term 在帧内的位置
struct TermLayout {
    int offset;  ///< 在帧内的起始偏移
    int dim;     ///< 该 term 占用的 float 数量
};

/**
 * @brief 观测段组装器抽象基类
 */
class ObsSegmentAssembler {
public:
    virtual ~ObsSegmentAssembler() = default;

    /**
     * @brief 初始化
     * @param frame_dim        单帧总维度
     * @param terms            各 term 在帧内的布局
     * @param history_length   历史帧数（仅 flat_history / frame_history 使用）
     * @param order            "oldest_first" 或 "newest_first"
     * @param include_current  是否在读取前写入当前帧
     */
    virtual void Init(int frame_dim,
                    const std::vector<TermLayout> &terms,
                    int history_length,
                    const std::string &order,
                    bool include_current) = 0;

    /// 该段输出的 float 总数
    virtual int OutputDim() const = 0;

    /**
     * @brief 组装输出
     * @param current_frame  当前帧数据（frame_dim 个 float）
     * @param out            输出缓冲区（至少 OutputDim() 个 float）
     */
    virtual void Assemble(const float *current_frame, float *out) = 0;
};

}  // namespace rl_policy

#endif  // OBS_ASSEMBLER_H
