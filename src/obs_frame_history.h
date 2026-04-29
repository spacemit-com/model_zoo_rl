/**
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file obs_frame_history.h
 * @brief 按帧滑窗历史缓冲组装器实现
 *
 * 对应 YAML 中 mode: frame_history 的观测段处理策略。
 * 按帧（完整观测）组织历史，适用于跨帧时间相关性重要的场景（如 LSTM）。
 */

#ifndef OBS_FRAME_HISTORY_H
#define OBS_FRAME_HISTORY_H

#include "obs_assembler.h"

#include <string>
#include <vector>

namespace rl_policy {

/**
 * @brief 按帧滑窗历史缓冲组装器
 *
 * 对应 YAML 中 mode: frame_history 的段。
 *
 * 输出布局（以 frame_dim=D、history_length=H 为例）:
 *   [frame_oldest(D), ..., frame_newest(D)]
 *
 * 每次 assemble 将整个窗口左移一帧，新帧放末尾。
 * 输出维度 = frame_dim × history_length
 */
class ObsFrameHistory : public ObsSegmentAssembler {
public:
    void Init(int frame_dim,
            const std::vector<TermLayout> &terms,
            int history_length,
            const std::string &order,
            bool include_current) override;
    int OutputDim() const override;
    void Assemble(const float *current_frame, float *out) override;

private:
    int frame_dim_ = 0;
    int history_length_ = 0;
    int total_size_ = 0;

    /// 滑窗缓冲区 [frame_dim * history_length]
    std::vector<float> slide_buf_;
};

}  // namespace rl_policy

#endif  // OBS_FRAME_HISTORY_H
