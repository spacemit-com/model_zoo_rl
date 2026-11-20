#!/usr/bin/env bash
# Regression checks using a policy configured with provider: cpu and an ONNX model.
set -euo pipefail
if [[ $# != 5 ]]; then
    echo "usage: $0 BUILD_DIR POLICY_YAML POLICY_NAME ROBOT_DIR VALID_ONNX" >&2
    exit 2
fi
build_dir=$(realpath "$1")
policy_yaml=$(realpath "$2")
policy_name=$3
robot_dir=$(realpath "$4")
model=$(realpath "$5")
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
"$build_dir/benchmark_policy_executor" "$policy_yaml" "$policy_name" "$robot_dir" \
    --warmup 1 --rounds 3 > "$work/default.log"
cat "$work/default.log"
grep -q "Provider:    requested=cpu" "$work/default.log"
"$build_dir/benchmark_policy_executor" "$policy_yaml" "$policy_name" "$robot_dir" \
    --provider cpu --warmup 1 --rounds 3
mkdir -p "$work/good" "$work/bad" "$work/mixed"
ln -s "$model" "$work/good/model.onnx"
ln -s "$model" "$work/mixed/model.onnx"
printf 'not an ONNX model\n' > "$work/bad/invalid.onnx"
cp "$work/bad/invalid.onnx" "$work/mixed/invalid.onnx"
"$build_dir/test_onnx_infer" --scan "$work/good"
for fixture in bad mixed; do
    if "$build_dir/test_onnx_infer" --scan "$work/$fixture"; then
        echo "scan incorrectly succeeded for $fixture" >&2
        exit 1
    fi
done
