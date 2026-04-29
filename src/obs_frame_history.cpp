/**
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file obs_frame_history.cpp
 * @brief 按帧的滑窗历史观测组装器实现
 */

#include "obs_frame_history.h"

#include <string>
#include <vector>
#include <cstring>
#include <stdexcept>

namespace rl_policy {

void ObsFrameHistory::Init(int frame_dim,
                            const std::vector<TermLayout> & /*terms*/,
                            int history_length,
                            const std::string & /*order*/,
                            bool /*include_current*/) {
    frame_dim_ = frame_dim;
    history_length_ = history_length;

    if (history_length_ <= 0) {
        throw std::runtime_error("[ObsFrameHistory] 需要正的 history length, 得到 " +
                                std::to_string(history_length_));
    }

    total_size_ = frame_dim_ * history_length_;
    slide_buf_.assign(total_size_, 0.0f);
}

int ObsFrameHistory::OutputDim() const {
    return total_size_;
}

void ObsFrameHistory::Assemble(const float *current_frame, float *out) {
    // 滑窗：左移一帧，新帧放末尾
    if (total_size_ > frame_dim_) {
        std::memmove(slide_buf_.data(),
                    slide_buf_.data() + frame_dim_,
                    (total_size_ - frame_dim_) * sizeof(float));
    }
    std::memcpy(
        slide_buf_.data() + total_size_ - frame_dim_, current_frame, frame_dim_ * sizeof(float));

    // 输出整个缓冲区
    std::memcpy(out, slide_buf_.data(), total_size_ * sizeof(float));
}

}  // namespace rl_policy
