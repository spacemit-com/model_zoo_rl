/**
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file obs_none.cpp
 * @brief 无历史观测组装器实现（仅输出当前帧）
 */

#include "obs_none.h"

#include <string>
#include <vector>
#include <cstring>

namespace rl_policy {

void ObsNone::Init(int frame_dim,
                    const std::vector<TermLayout> & /*terms*/,
                    int /*history_length*/,
                    const std::string & /*order*/,
                    bool /*include_current*/) {
    frame_dim_ = frame_dim;
}

int ObsNone::OutputDim() const {
    return frame_dim_;
}

void ObsNone::Assemble(const float *current_frame, float *out) {
    std::memcpy(out, current_frame, frame_dim_ * sizeof(float));
}

}  // namespace rl_policy
