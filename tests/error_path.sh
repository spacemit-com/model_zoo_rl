#!/usr/bin/env bash
# rl error-path 用例：配置加载 / 策略选择的非法输入须快速失败。
# 全部不依赖 onnx 模型（失败发生在 LoadPolicyConfigFromYaml 解析/校验阶段）。
# CWD = 模块根（components/model_zoo/rl），test_policy_executor 经 staging/bin 在 PATH。
set -uo pipefail

fail=0
CFG=example/config_example.yaml

expect_fail() {  # $1=场景描述, 其余=命令
  local desc="$1"; shift
  if timeout 15 "$@" >/dev/null 2>&1; then
    echo "[FAIL] 期望非 0 却成功（$desc）: $*"; fail=1
  else
    local rc=$?
    if [[ "$rc" -eq 124 ]]; then echo "[FAIL] hang（$desc）: $*"; fail=1
    else echo "[OK] 正确拒绝(rc=$rc, $desc)"; fi
  fi
}

expect_fail "缺参"           test_policy_executor
expect_fail "坏 yaml 路径"   test_policy_executor /nonexistent.yaml motion /tmp
expect_fail "不存在的策略名" test_policy_executor "$CFG" __no_such_policy__ /tmp

if [[ "$fail" -ne 0 ]]; then echo "rl error-path: FAILED"; exit 1; fi
echo "rl error-path: PASS"
