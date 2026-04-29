# rl — RL 策略推理模块

## 项目简介

RL 策略推理执行器，负责 YAML 配置解析、观测组装、ONNX 推理与动作映射。模块设计与机器人型号无关，完全由 YAML 驱动。

## 功能特性

**支持：**
- MLP / LSTM / obs_hist 三种模型结构，自动识别
- 三种观测历史模式：无历史、flat_history（按变量分组）、frame_history（按帧滑窗）
- 运行时动态策略切换（重新加载 YAML 即可）
- 自定义标量注入（如相位 `z`、标志 `stand_flag`）
- 动作映射：scale、blend、default_pos 全部由 YAML 配置

**不支持：**
- PyTorch 原生推理（需先转换为 ONNX）
- aarch64 / RISC-V 上的 GPU 加速推理

## 快速开始

### 环境准备

**PC 端（x86_64）**：

```bash
# 系统依赖
sudo apt install -y libeigen3-dev libyaml-cpp-dev cmake g++

# ONNX Runtime 1.21.0（仅需 x86_64 本机 RL 推理时安装）
wget https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-x64-1.21.0.tgz
tar -xzf onnxruntime-linux-x64-1.21.0.tgz
sudo cp -r onnxruntime-linux-x64-1.21.0/include/* /usr/local/include/
sudo cp -r onnxruntime-linux-x64-1.21.0/lib/* /usr/local/lib/
sudo ldconfig
```

> CMake 查找顺序：`ONNXRUNTIME_DIR` 环境变量或编译参数 → `/usr/local` → `/usr`。若安装到非默认路径，可通过 `export ONNXRUNTIME_DIR=/path/to/onnxruntime` 或编译时 `-DONNXRUNTIME_DIR=/path/to/onnxruntime` 指定。其他版本（≥ 1.17）见 [github.com/microsoft/onnxruntime/releases](https://github.com/microsoft/onnxruntime/releases)。

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

编译产物安装至 `output/staging/`：
- 动态库：`output/staging/lib/librl.so`
- 测试程序：`output/staging/bin/test_policy_executor`、`output/staging/bin/test_onnx_infer`
- 基准工具：`output/staging/bin/benchmark_policy_executor`、`output/staging/bin/benchmark_onnx_infer`
- 基准脚本：`output/staging/bin/run_benchmark_policy.sh`、`output/staging/bin/run_benchmark_onnx.sh`

**独立 cmake 编译**：

```bash
cd components/model_zoo/rl
mkdir build && cd build
cmake ..
make
```

### 运行示例

**功能测试：**

```bash
cd ~/spacemit_robot
./output/staging/bin/test_policy_executor \
  application/native/humanoid_unitree_g1/config/g1.yaml motion \
  application/native/humanoid_unitree_g1
```

**性能基准测试：**

两个基准工具，目标不同，按需选用：

`run_benchmark_onnx.sh` — 纯推理基准，隔离测试 ONNX Runtime 推理引擎本身的延迟，排除 obs 组装和动作映射的干扰。适用于对比不同硬件（x86 vs K3）或不同模型结构（MLP / LSTM）的推理性能。

`run_benchmark_policy.sh` — 完整链路基准，测试真实运控场景的端到端性能，对 `AssembleObs → Infer → MapActionToTargetPos` 三个阶段分别计时，识别性能瓶颈。

```bash
cd ~/spacemit_robot/output/staging/bin
run_benchmark_onnx.sh                               # g1 + motion，默认参数
run_benchmark_policy.sh g1 motion --rounds 2000     # 增加测试轮数
run_benchmark_policy.sh tiangong walk               # 其他机型/策略
```

公共参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `robot` | 机型名称（g1/go1/h1_2/r1/asimov/tinker/qinglong/tiangong） | `g1` |
| `policy` | 策略名称 | `motion` |
| `--warmup N` | 预热轮数 | `100` |
| `--rounds N` | 正式统计轮数 | `1000` |
| `--hz N` | 目标控制频率，用于计算超时率 | `50` |
| `--verbose` | 逐帧输出延迟 | 关闭 |

输出指标（Avg / Std / P50 / P95 / P99 / Max / Miss）含义见详细使用章节。

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
| `Init` | `const PolicyExecutorConfig &cfg` | 初始化执行器：加载 ONNX 模型、初始化观测处理、验证维度 |
| `ObsDim` | `void → int` | 返回预期观测向量维度 |
| `ActionDim` | `void → int` | 返回动作向量维度 |
| `HasLstm` | `void → bool` | 模型是否包含 LSTM 单元（影响推理循环方式） |
| `HasObsHist` | `void → bool` | 是否使用 obs_hist 输入（长期观测历史） |
| `SetCustomScalar / GetCustomScalar` | `const std::string &name, float value` | 设置/获取自定义标量（如 `"z"` 相位、`"stand_flag"` 标志） |
| `AssembleObs` | 传感器数据 → `Eigen::VectorXf &out_obs` | 组装观测向量：计算各段、交给对应处理器、拼接输出 |
| `Infer` | `const Eigen::VectorXf &obs` → `std::vector<double> &action` | 执行推理：MLP / LSTM / obs_hist 均自动处理 |
| `MapActionToTargetPos` | `const std::vector<double> &action` → `std::vector<double> &target_pos` | 将策略动作映射为全身关节目标位置 |

#### 核心数据结构

**`PolicyExecutorConfig`** — 策略执行参数：
- 模型路径、动作映射（scale、blend、default_pos）
- 段式观测配置（terms、mode、length、order、include_current）
- 观测归一化参数（ang_vel_scale、dof_pos_scale 等）
- phase / gait_phase 相位参数
- 自定义标量默认值

**`LoadedPolicyConfig`** — YAML 解析结果：
- `exec_cfg` — PolicyExecutorConfig
- `command_init` — 初始速度指令 [vx, vy, wz]
- `infer_decimation` — 推理降频参数
- `max_roll / max_pitch` — 安全约束
- `thread_cpu_id / thread_priority` — 线程配置

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
| **无历史** | 省略 `mode` 或 `mode: ""` | 输出当前帧的 obs terms | G1 motion、Tinker trot |
| **flat_history** | `mode: flat_history` | 按变量分组的环形历史缓冲，同一 term 的多帧数据连续排列 | G1 dance/kungfu、青龙结构化历史 |
| **frame_history** | `mode: frame_history` | 按帧的滑窗历史缓冲，每帧完整的 obs 连续排列 | 天工 walk、青龙滑窗历史 |

#### 注意事项

1. **观测维度匹配**：AssembleObs 输出维度必须等于 ObsDim()；若 strict_obs_dim_check = true，Infer 会严格验证
2. **LSTM 推理循环**：若 HasLstm() = true，需在多帧推理中保持执行器状态，不可销毁/重建
3. **自定义标量生命周期**：SetCustomScalar 后有效直至下次修改；建议在每个推理循环的 AssembleObs 前重新设置
4. **运行时策略切换**：调用 `LoadPolicyConfigFromYaml(yaml_path, new_policy_name, robot_dir)` 加载新策略后，销毁旧 PolicyExecutor，创建新实例并 Init；FSM 负责管理切换时机
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
