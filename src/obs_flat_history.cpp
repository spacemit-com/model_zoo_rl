/**
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file obs_flat_history.cpp
 * @brief 按变量分组的环形历史观测组装器实现
 */

#include "obs_flat_history.h"

#include <string>
#include <vector>
#include <stdexcept>

namespace rl_policy {

void ObsFlatHistory::Init(int frame_dim,
                        const std::vector<TermLayout> &terms,
                        int history_length,
                        const std::string &order,
                        bool include_current) {
    frame_dim_ = frame_dim;
    terms_ = terms;
    history_length_ = history_length;
    newest_first_ = (order == "newest_first");
    include_current_ = include_current;

    if (history_length_ <= 0) {
        throw std::runtime_error("[ObsFlatHistory] 需要正的 history length, 得到 " +
                                std::to_string(history_length_));
    }

    ring_buf_.resize(history_length_, std::vector<float>(frame_dim_, 0.0f));
    ring_idx_ = 0;
}

int ObsFlatHistory::OutputDim() const {
    return frame_dim_ * history_length_;
}

void ObsFlatHistory::Assemble(const float *current_frame, float *out) {
    const int H = history_length_;

    // include_current=true: 先写入当前帧到环形缓冲，再读取
    if (include_current_) {
        auto &slot = ring_buf_[ring_idx_];
        for (int i = 0; i < frame_dim_; ++i)
            slot[i] = current_frame[i];
        ring_idx_ = (ring_idx_ + 1) % H;
    }

    // 输出：按变量分组，每组内按时间排列
    int out_idx = 0;
    for (const auto &tl : terms_) {
        for (int h = 0; h < H; ++h) {
            int frame_idx;
            if (newest_first_) {
                // newest first: ring_idx_-1 是最新
                frame_idx = (ring_idx_ - 1 - h + H) % H;
            } else {
                // oldest first: ring_idx_ 是最旧
                frame_idx = (ring_idx_ + h) % H;
            }
            const auto &frame = ring_buf_[frame_idx];
            for (int d = 0; d < tl.dim; ++d) {
                out[out_idx++] = frame[tl.offset + d];
            }
        }
    }

    // include_current=false: 先读取历史，再写入当前帧
    if (!include_current_) {
        auto &slot = ring_buf_[ring_idx_];
        for (int i = 0; i < frame_dim_; ++i)
            slot[i] = current_frame[i];
        ring_idx_ = (ring_idx_ + 1) % H;
    }
}

}  // namespace rl_policy
