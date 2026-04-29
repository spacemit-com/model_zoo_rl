#!/bin/bash
# 完整链路性能基准：AssembleObs → Infer → MapActionToTargetPos 分段延迟
#
# 用法:
#   run_benchmark_policy.sh [robot] [policy] [--warmup N] [--rounds N] [--verbose]
#
# 示例:
#   run_benchmark_policy.sh                          # 默认 g1 + motion
#   run_benchmark_policy.sh g1 motion --rounds 500

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
: "${SDK_ROOT:=$(cd "$SCRIPT_DIR/../../.." && pwd)}"

ROBOT="${1:-g1}"
POLICY="${2:-motion}"
shift 2 2>/dev/null || shift $# 2>/dev/null

# 宇树系列机型目录名带 unitree_ 前缀
case "$ROBOT" in
    g1|go1|h1_2|r1) ROBOT_DIR_NAME="unitree_${ROBOT}" ;;
    *)               ROBOT_DIR_NAME="$ROBOT" ;;
esac

YAML="$SDK_ROOT/application/native/humanoid_${ROBOT_DIR_NAME}/config/${ROBOT}.yaml"
ROBOT_DIR="$SDK_ROOT/application/native/humanoid_${ROBOT_DIR_NAME}"

if [ ! -f "$YAML" ]; then
    echo "[run_benchmark_policy] 配置文件不存在: $YAML" >&2
    echo "  支持机型: g1 h1_2 r1 go1 asimov tinker qinglong tiangong" >&2
    exit 1
fi

exec "$SCRIPT_DIR/benchmark_policy_executor" "$YAML" "$POLICY" "$ROBOT_DIR" "$@"
