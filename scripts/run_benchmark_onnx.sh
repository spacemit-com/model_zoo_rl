#!/bin/bash
# synthetic ONNX backend benchmark；不代表机器人完整运行链路。
#
# 用法:
#   run_benchmark_onnx.sh <robot> <policy> [benchmark options]
#
# 示例:
#   run_benchmark_onnx.sh g1 walk_mjlab --provider cpu --threads 2 \
#       --affinity 0,1 --mode periodic --hz 50 --rounds 1000 --csv /tmp/g1.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
: "${SDK_ROOT:=$(cd "$SCRIPT_DIR/../../.." && pwd)}"

if [ "$#" -lt 2 ] || [[ "$1" == --* ]] || [[ "$2" == --* ]]; then
    echo "用法: run_benchmark_onnx.sh <robot> <policy> [benchmark options]" >&2
    exit 2
fi
ROBOT="$1"
POLICY="$2"
shift 2

DIRECT_DIR="$SDK_ROOT/application/native/humanoid_${ROBOT}"
UNITREE_DIR="$SDK_ROOT/application/native/humanoid_unitree_${ROBOT}"
if [ -f "$DIRECT_DIR/config/${ROBOT}.yaml" ]; then
    ROBOT_DIR="$DIRECT_DIR"
elif [ -f "$UNITREE_DIR/config/${ROBOT}.yaml" ]; then
    ROBOT_DIR="$UNITREE_DIR"
else
    echo "[run_benchmark_onnx] 找不到机型配置，已检查:" >&2
    echo "  $DIRECT_DIR/config/${ROBOT}.yaml" >&2
    echo "  $UNITREE_DIR/config/${ROBOT}.yaml" >&2
    exit 1
fi
YAML="$ROBOT_DIR/config/${ROBOT}.yaml"

exec "$SCRIPT_DIR/benchmark_onnx_infer" "$YAML" "$POLICY" "$ROBOT_DIR" "$@"
