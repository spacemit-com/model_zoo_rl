/**
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file obs_flat_history.h
 * @brief 按变量分组的环形历史缓冲组装器实现
 *
 * 对应 YAML 中 mode: flat_history 的观测段处理策略。
 * 按变量（term）分组组织历史，适用于变量相关性强的场景。
 */

#ifndef OBS_FLAT_HISTORY_H
#define OBS_FLAT_HISTORY_H

#include "obs_assembler.h"

#include <string>
#include <vector>

namespace rl_policy {

/**
 * @brief 按变量分组的环形历史缓冲组装器
 *
 * 对应 YAML 中 mode: flat_history 的段。
 *
 * 输出布局（以 3 个 term、history_length=H 为例）:
 *   [term0_frame0, term0_frame1, ..., term0_frameH-1,
 *    term1_frame0, term1_frame1, ..., term1_frameH-1,
 *    term2_frame0, term2_frame1, ..., term2_frameH-1]
 *
 * 输出维度 = per_frame_dim × history_length
 */
class ObsFlatHistory : public ObsSegmentAssembler {
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
    bool newest_first_ = false;
    bool include_current_ = true;
    std::vector<TermLayout> terms_;

    /// 环形缓冲区 [history_length][frame_dim]
    std::vector<std::vector<float>> ring_buf_;
    int ring_idx_ = 0;
};

}  // namespace rl_policy

#endif  // OBS_FLAT_HISTORY_H
