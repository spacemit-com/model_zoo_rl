#!/usr/bin/env bash
# rl inference 冒烟（scheduled，需真模型）。
# CWD = SDK 根（test.yaml 里 workdir: sdk）。
# 模型来源优先级：
#   1. 环境变量 SROBOTIS_RL_TEST_ROBOT_DIR / _CONFIG / _POLICY 显式指定
#   2. 自动探测 application/native/humanoid_unitree_g1（其 policy 目录有 onnx 时）
# 找不到模型 -> 非 0（在配置了模型的 scheduled 环境中视为环境未就绪）。
set -uo pipefail

ROBOT_DIR=${SROBOTIS_RL_TEST_ROBOT_DIR:-application/native/humanoid_unitree_g1}
CONFIG=${SROBOTIS_RL_TEST_CONFIG:-$ROBOT_DIR/config/g1.yaml}
POLICY=${SROBOTIS_RL_TEST_POLICY:-motion}

if [[ ! -f "$CONFIG" ]]; then
  echo "[SKIP/FAIL] 配置不存在: $CONFIG（设 SROBOTIS_RL_TEST_CONFIG 指定）"; exit 1
fi
if ! ls "$ROBOT_DIR"/policy/**/*.onnx >/dev/null 2>&1 && ! ls "$ROBOT_DIR"/policy/*.onnx >/dev/null 2>&1; then
  echo "[FAIL] $ROBOT_DIR/policy 下无 onnx 模型；先 download_humanoid_policy 或设 SROBOTIS_RL_TEST_ROBOT_DIR"; exit 1
fi

out=$(timeout 120 test_policy_executor "$CONFIG" "$POLICY" "$ROBOT_DIR" 2>&1) || {
  echo "[FAIL] test_policy_executor 退出非 0"; echo "$out" | tail -15; exit 1; }

# 断言：维度全部匹配（demo 在不匹配时打印「✗ 警告」）+ 跑到成功标志
if grep -q "✗ 警告" <<<"$out"; then echo "[FAIL] 维度不匹配："; grep "✗" <<<"$out"; exit 1; fi
if ! grep -q "所有测试成功" <<<"$out"; then echo "[FAIL] 未跑到成功标志"; echo "$out" | tail -15; exit 1; fi

echo "rl inference-smoke: PASS ($ROBOT_DIR / $POLICY)"
