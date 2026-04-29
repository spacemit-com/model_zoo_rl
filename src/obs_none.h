/**
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file obs_none.h
 * @brief 无历史观测组装器实现
 *
 * 对应 YAML 中不指定 mode 或 mode 为空的观测段处理策略。
 * 直接输出当前帧观测，不维护历史缓冲。
 */

#ifndef OBS_NONE_H
#define OBS_NONE_H

#include "obs_assembler.h"

#include <string>
#include <vector>

namespace rl_policy {

/**
 * @brief 无历史组装器 —— 直接输出当前帧
 *
 * 对应 YAML 中不指定 mode 或 mode 为空的段。
 */
class ObsNone : public ObsSegmentAssembler {
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
};

}  // namespace rl_policy

#endif  // OBS_NONE_H
