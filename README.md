# rl — RL 策略推理模块

## 项目简介

RL 策略推理执行器，负责 YAML 配置解析、观测组装、ONNX 推理与动作映射。模块设计与机器人型号无关，完全由 YAML 驱动。

## 功能特性

**支持：**
- 按 ONNX 张量名声明 `model_io`，不按 MLP/LSTM 等模型类别分派
- 输入源：`observation` / `observation_history` / `feedback` / `constant` / `external`
- 输出目标：`action` / `expose` / `ignore`；未知 I/O 一律初始化失败
- 任意数量和命名的反馈状态，observation history 与 feedback 可作为独立输入同时使用
- 每个策略必须显式声明完整 `model_io`，不根据 tensor 名或 shape 猜测语义
- 三种观测历史模式：无历史、flat_history（按变量分组）、frame_history（按帧滑窗）
- 运行时动态策略切换（重新加载 YAML 即可）
- 自定义 observation term 和独立 ONNX 输入/输出的上层注入、读取
- observation/action clip，以及 scale、blend、default_pos 动作映射

**不支持：**
- PyTorch 原生推理（需先转换为 ONNX）
- ONNX `string` / `complex` 外部 I/O、静态 batch>1、运行时改变 shape
- FP8/INT4 的 float 语义自动转换；这类类型仅支持原生 packed `TensorView`
- 多个 action head、随机分布采样和非关节位置 action 语义
- aarch64 / RISC-V 上的 GPU 加速推理

## 快速开始

### 环境准备

**PC 端（x86_64）**：

```bash
# 系统依赖
sudo apt install -y libeigen3-dev libyaml-cpp-dev cmake g++
```

ONNX Runtime 由 CMake 处理。CMake 按以下顺序查找，命中即用：

1. `-DONNXRUNTIME_DIR=...` 编译参数
2. 环境变量 `ONNXRUNTIME_DIR`
3. 系统路径 `/usr/local`、`/usr`
4. 缓存路径 `~/.cache/thirdparty/onnxruntime/onnxruntime-linux-x64-1.21.0/`
5. 上述均未命中：从官方 release 拉取 `onnxruntime-linux-x64-1.21.0.tgz` 解压到第 4 项缓存路径

> 离线/受限网络环境可设 `SROBOTIS_THIRDPARTY_FETCH_OFF=ON` 禁用第 5 步拉取，并通过
> `export ONNXRUNTIME_DIR=/path/to/onnxruntime` 指向预先下载好的目录。其他版本（≥ 1.17）见
> [github.com/microsoft/onnxruntime/releases](https://github.com/microsoft/onnxruntime/releases)。

**K3 板卡端**：

```bash
# 系统依赖
sudo apt install -y libeigen3-dev libyaml-cpp-dev spacemit-tcm pkg-config

# SpacemiT 定制版 ONNX Runtime（含 A100 核 EP 加速）
# 如已安装标准版，先卸载：
sudo apt remove libonnxruntime-dev libonnxruntime1.23 python3-onnxruntime
# 安装定制版：
sudo apt install -y libonnx-dev libonnx-testdata libonnx1t64 \
  libonnxruntime-providers onnxruntime-tools python3-onnx \
  python3-spacemit-ort spacemit-onnxruntime
```

### 构建编译

**SDK 内编译（mm）**：

```bash
source ~/spacemit_robot/build/envsetup.sh
cd components/model_zoo/rl
mm
```

单配置生成器未显式指定 `CMAKE_BUILD_TYPE` 时，本组件默认使用 `Release`
（`-O3 -DNDEBUG`）。benchmark 启动时会打印实际 build type 和编译 flags；
不要使用旧的 O0 产物做性能结论。

编译产物安装至 `output/staging/`：
- 动态库：`output/staging/lib/librl.so`
- 测试程序：`output/staging/bin/test_policy_executor`、`output/staging/bin/test_onnx_infer`
- 基准工具：`output/staging/bin/benchmark_policy_executor`、`output/staging/bin/benchmark_onnx_infer`
- 基准脚本：`output/staging/bin/run_benchmark_policy.sh`、`output/staging/bin/run_benchmark_onnx.sh`
- 辅助脚本：`scripts/extract_packed_motion.py` — 从包装版 ONNX 提取 motion 数据为 npz（依赖系统 Python 的 onnx + numpy）

**独立 cmake 编译**：

```bash
cmake -S components/model_zoo/rl -B /tmp/rl-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/rl-build -j
```

### 运行示例

**功能测试：**

```bash
cd ~/spacemit_robot
./output/staging/bin/test_policy_executor \
  application/native/humanoid_unitree_g1/config/g1.yaml walk_mjlab \
  application/native/humanoid_unitree_g1
```

**性能基准测试：**

组件提供两个性能工具，分别隔离 backend 和 `PolicyExecutor`：

`run_benchmark_onnx.sh` 使用 synthetic 输入测量当前 ONNX backend。它会按
`model_io` 回灌 recurrent feedback、应用 constant；缺少默认值的 external
输入会使用 synthetic 数据并明确列出。指标分为 `ONNX Run` 和包含输入更新、
feedback copy 的 `Backend step`。

`run_benchmark_policy.sh` 测量
`AssembleObs → Infer → MapActionToTargetPos`，输入来自 YAML 配置和固定的
synthetic robot state。缺少默认值的 external model input 或 custom array
无法由该工具生成，因此会在计时前明确拒绝；这类策略需要由能够提供完整输入的
集成或 replay runner 测量。

```bash
cd ~/spacemit_robot/output/staging/bin
./run_benchmark_policy.sh g1 walk_mjlab \
  --provider cpu --threads 2 --affinity 0,1 \
  --ort-spinning off \
  --mode periodic --hz 50 --rounds 1000 --csv /tmp/g1_walk_mjlab.csv

./run_benchmark_onnx.sh g1 walk_mjlab \
  --provider cpu --threads 2 --affinity 0,1 \
  --mode throughput --rounds 1000 --csv /tmp/g1_walk_mjlab_backend.csv
```

公共参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `robot` | 机型名称；wrapper 自动检查 `humanoid_<robot>` 与 `humanoid_unitree_<robot>` | 必填 |
| `policy` | YAML 中的策略名称 | 必填 |
| `--warmup N` | 预热轮数 | `100` |
| `--rounds N` | 正式统计轮数 | `1000` |
| `--mode throughput\|periodic` | 背靠背吞吐或绝对时钟周期释放 | `throughput` |
| `--hz N` | periodic 频率；未覆盖时读取 YAML `rl_policy.rl_dt` | YAML |
| `--overrun drop\|backlog` | periodic 过载时丢过期 release 或保留 backlog | `drop` |
| `--provider auto\|cpu\|spacemit` | ONNX provider；显式 spacemit 初始化失败会退出 | `auto` |
| `--threads N` | CPU intra-op 或 SpaceMIT EP 线程数 | `1` |
| `--ort-spinning on\|off` | 是否允许 ORT intra-op worker 在无任务时 busy-spin；用于周期负载 A/B | `on` |
| `--affinity CPU_LIST` | 进程 affinity，例如 `0`、`0,1`；传给 EP 时自动转为官方分号格式 | 当前进程 mask |
| `--ep-dump-subgraphs` | 导出 EP 实际编译子图，用于确认不是全量回退 CPU | 关闭 |
| `--ep-profile PREFIX` | 导出 EP 执行 profile JSON | 关闭 |
| `--csv PATH` | 保存逐轮原始数据 | 不保存 |
| `--verbose` | 计时结束后输出逐轮数据，不干扰计时区间 | 关闭 |

`throughput` 模式只报告 service time/吞吐，不输出实时满足结论。`periodic`
使用 `steady_clock::sleep_until` 绝对时间释放，记录 release jitter、service、
release-to-done response、deadline lateness、miss、drop 和 backlog。RL 组件内没有
真实传感器采样与电机下发时间戳，因此这里不能给出整机 action age。即使 periodic
无 miss，也只证明当前 benchmark 边界，不证明整机实时性。P99.99 至少使用
100000 轮，并必须通过 `--csv` 保留原始样本。

`--ort-spinning off` 对应 ONNX Runtime 的
`session.intra_op.allow_spinning=0`。它只控制 ORT intra-op 线程池，不控制
SpaceMIT EP 自己的 worker；provider 为 `spacemit` 时不能据此推断 EP 已停止
busy-spin。关闭后通常能降低周期调用之间的 CPU 占用，但可能增加 worker 唤醒
延迟，因此报告必须同时比较 periodic 的 response/p99/max 与进程 CPU、调度等待。
stdout 和 CSV 都会记录实际配置。

### CI 测试

模块自带 `test.yaml`（CI 用例清单）+ `tests/`，经 SDK 根目录的 `robot-test` 运行：

```bash
scripts/test/robot-test list components/model_zoo/rl
scripts/test/robot-test run  components/model_zoo/rl --scope pr        # 配置加载/错误路径（不依赖模型）
scripts/test/robot-test run  components/model_zoo/rl --scope scheduled # 真模型推理冒烟（需已下载 policy）
```

仓库内无 onnx 模型，故 PR 档只验配置/错误处理；推理冒烟归 scheduled。

## 详细使用

### 接口说明

本模块对外接口分为配置解析、策略执行和数据结构三大类。
各接口的详细使用范例和完整测试，请参考：**[example/test_policy_executor.cpp](example/test_policy_executor.cpp)**

#### 配置加载接口

| 接口名称 | 参数类型 | 返回值 | 功能说明 |
| :--- | :--- | :--- | :--- |
| `LoadPolicyConfigFromYaml` | `yaml_path, policy_name, robot_dir` | `LoadedPolicyConfig` | 按策略名加载指定策略配置，支持运行时动态切换；`robot_dir` 为机器人资源根目录绝对路径，用于解析 model_path 等相对路径 |

#### `PolicyExecutor` 策略执行器

| 接口名称 | 参数 / 返回 | 功能说明 |
| :--- | :--- | :--- |
| `Init` | `const PolicyExecutorConfig &cfg` | 加载 ONNX，建立并严格校验全部 named tensor binding |
| `ObsDim` | `void → int` | 返回预期观测向量维度 |
| `ActionDim` | `void → int` | 返回动作向量维度 |
| `FeedbackStateCount` | `void → int` | 自动回灌的 feedback 张量对数量 |
| `HasObsHist` | `void → bool` | 是否使用 `observation_history` 输入 |
| `SetCustomScalar / GetCustomScalar` | `const std::string &name, float value` | 设置/获取自定义标量（如 `"z"` 相位、`"stand_flag"` 标志） |
| `SetCustomArray` | `const std::string &name, const float *data, int size` | 推入 N 维自定义数组 obs term（泛型扩展点）。配套 yaml `custom_array_dims: {name: dim}` 声明维度。例：tracking 策略通过 `SetCustomArray("motion_command", buf, 58)` 注入参考关节数据 |
| `SetModelInput` | `key, float data, size` 或 `key, TensorView` | 以 float 语义值或原生 dtype 设置 `source: external` 输入 |
| `GetModelOutput` | `key → const std::vector<float> &` | 以 float 便捷视图读取 `target: expose` 输出 |
| `GetModelOutputTensor` | `key → TensorView` | 读取保留原生 dtype 的 `target: expose` 输出 |
| `AssembleObs` | 传感器数据 → `Eigen::VectorXf &out_obs` | 组装观测向量：计算各段、交给对应处理器、拼接输出 |
| `Infer` | `const Eigen::VectorXf &obs` → `std::vector<double> &action` | 按声明绑定输入、执行推理、反馈状态并处理输出 |
| `MapActionToTargetPos` | `const std::vector<double> &action` → `std::vector<double> &target_pos` | 将策略动作映射为全身关节目标位置 |

#### 核心数据结构

**`PolicyExecutorConfig`** — 策略执行参数：
- 模型路径、动作映射（scale、blend、default_pos）
- 完整 `model_io` 输入输出绑定、可选 observation/action clip
- 段式观测配置（terms、mode、length、order、include_current）
- 观测归一化参数（ang_vel_scale、dof_pos_scale 等）
- phase / gait_phase 相位参数
- 自定义标量默认值

**`LoadedPolicyConfig`** — YAML 解析结果：
- `exec_cfg` — PolicyExecutorConfig
- `command_init` — 初始速度指令 [vx, vy, wz]
- `kp / kd` — 该策略训练时的 PD 增益（每策略独立，进入 RL 后下发给驱动）
- `infer_decimation` — 推理降频参数
- `max_roll / max_pitch` — 安全约束

**`ObsSegmentConfig`** — 观测段配置：
- `terms` — 观测项列表（如 base_gyro、joint_pos、phase）
- `mode` — 处理模式：`""` (无历史) / `"flat_history"` / `"frame_history"`
- `length / order / include_current` — 历史相关参数

#### 集成步骤

```cpp
#include "rl_service.h"
using namespace rl_policy;

// 1) 从 YAML 加载配置
LoadedPolicyConfig loaded_cfg = LoadPolicyConfigFromYaml(yaml_path, policy_name, robot_dir);

// 2) 初始化执行器
PolicyExecutor policy;
policy.Init(loaded_cfg.exec_cfg);

// 3) 观测组装（每帧）
Eigen::VectorXf obs;
policy.AssembleObs(gyro, rpy, cmd_vx, cmd_vy, cmd_wz,
                   joint_pos, joint_vel, base_quat, base_vel, dt, obs);

// 4) 推理
std::vector<double> action;
policy.Infer(obs, action);

// 5) 映射至目标位置
std::vector<double> target_pos;
policy.MapActionToTargetPos(action, target_pos);
```

#### 观测历史模式说明

| 模式 | YAML 值 | 说明 | 典型场景 |
|------|---------|------|----------|
| **无历史** | 省略 `mode` 或 `mode: ""` | 输出当前帧的 obs terms | G1 walk_rlgym、Tinker trot |
| **flat_history** | `mode: flat_history` | 按变量分组的环形历史缓冲，同一 term 的多帧数据连续排列 | G1 dance/kungfu、青龙结构化历史 |
| **frame_history** | `mode: frame_history` | 按帧的滑窗历史缓冲，每帧完整的 obs 连续排列 | 天工 walk、青龙滑窗历史 |

#### 模型 I/O 绑定

每个策略必须显式声明所有 ONNX 输入输出。以下配置同时演示 feedback、constant、external、expose 和 ignore：

```yaml
model_io:
  inputs:
    - {name: obs, source: observation}
    - {name: h0, source: feedback, output: hn}
    - {name: time_step, source: constant, value: 0.0}
    - {name: mask, source: external, key: policy_mask, default: [1.0]}
  outputs:
    - {name: actions, target: action}
    - {name: value, target: expose, key: critic_value}
    - {name: debug, target: ignore}
clip_observations: 100
clip_actions: 100
```

`observation` 可用 `offset` 把完整观测切给多个输入；`observation_history` 用 `history_of` 指定要跟踪的 `observation` 输入，不跟踪 feedback、constant 或 external。`feedback` 初始为零，之后每帧自动回灌。feedback 输入输出允许 shape 不同，但必须具有相同 dtype 和元素数，复制时按扁平连续缓冲区重解释。`external` 无 `default` 时，首次 `Infer` 前必须由上层调用 `SetModelInput`。绑定缺失、重复绑定，以及自动 float 语义路径不支持的 dtype 会在 `Init` 失败；`external` 的原生 dtype 和元素数在 `SetModelInput` 时校验。

当前部署 ABI 固定为 batch=1，并要求恰好一个确定性关节位置 action 输出。模型中的符号维度会在初始化时固定为 1，不支持运行时改变 shape；明确声明为静态 batch>1 的 observation/action 会初始化失败。ONNX backend 由 `Ort::Value` 按模型声明持有原生 dtype；observation/action 可继续使用 float 便捷视图，`external` / `expose` 可通过 `TensorView` 保留 INT64 等真实类型和精度。FP8/INT4 支持原始 packed buffer，不提供无量化参数的 float 自动转换。`external` / `expose` 不替代 observation term；随机策略、多 action head 或力矩 action 需要增加对应的可复用输出语义，或在导出 ONNX 时封装成该 ABI。

#### 注意事项

1. **观测维度匹配**：AssembleObs 输出维度必须等于 ObsDim()；若 strict_obs_dim_check = true，Infer 会严格验证
2. **反馈状态生命周期**：若 FeedbackStateCount() > 0，需在多帧推理中保持执行器实例；重建执行器会把状态清零
3. **自定义标量生命周期**：SetCustomScalar 后有效直至下次修改；建议在每个推理循环的 AssembleObs 前重新设置
4. **运行时策略切换**：调用 `LoadPolicyConfigFromYaml(yaml_path, new_policy_name, robot_dir)` 加载新策略后，销毁旧 PolicyExecutor，创建新实例并 Init；调用方负责管理切换时机
5. **段式观测细节**：mode 为空表示无历史仅输出当前帧；flat_history 按变量分组；frame_history 按帧组织

## 常见问题

**Q：运行时报 `配置文件不存在`？**
确认 `robot` 参数与 `application/native/` 下的目录名对应：宇树系列传 `g1`/`go1`/`h1_2`/`r1`，其余传 `asimov`/`tinker`/`qinglong`/`tiangong`。

**Q：ONNX Runtime 找不到？**
确认已安装 ONNX Runtime 并放置在 `/usr` 或 `/usr/local` 下，或设置 `ONNXRUNTIME_ROOT` 环境变量指向安装目录。

**Q：推理维度不匹配？**
检查 YAML 中 `obs_segments` 的 terms 列表与模型实际输入维度是否一致，可通过 `ObsDim()` 打印预期维度排查。

## 版本与发布

变更记录见 git log。

## 贡献方式

贡献者与维护者名单见：`CONTRIBUTORS.md`

## License

本组件源码文件头声明为 Apache-2.0，最终以本目录 `LICENSE` 文件为准。
