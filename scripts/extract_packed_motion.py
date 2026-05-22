# Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
从包装版 ONNX 提取 motion 参考数据为 npz 格式。
自动发现 ONNX initializer 中的 motion 数据，支持 body 列表重映射。
"""

import argparse
import os
import re
import sys

import numpy as np

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    print("Error: onnx package required. Install: pip install onnx", file=sys.stderr)
    sys.exit(1)

WEIGHT_PATTERN = re.compile(r"\.(weight|bias|running_mean|running_var)(\.\d+)?$")


def get_metadata(model, key):
    for p in model.metadata_props:
        if p.key == key:
            return p.value
    return None


def strip_numeric_suffix(name):
    """去掉 ONNX 导出时附加的数字后缀（如 joint_pos.1 → joint_pos）。"""
    m = re.match(r"^(.+)\.\d+$", name)
    return m.group(1) if m else name


def is_network_weight(name):
    return bool(WEIGHT_PATTERN.search(name))


def discover_motion_initializers(model, min_frames=10):
    """从 ONNX initializer 中自动发现 motion 数据。

    识别标准：
    - 2D/3D 数组且第一维（帧数）>= min_frames
    - 名字不匹配网络权重模式（weight/bias/running_mean/running_var）
    """
    results = {}
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.ndim < 2 or arr.shape[0] < min_frames:
            continue
        name = strip_numeric_suffix(init.name)
        if name in results:
            continue
        if is_network_weight(init.name):
            continue
        results[name] = arr
    return results


def build_body_mapping(packed_names, full_names):
    mapping = {}
    unmatched = []
    for i, name in enumerate(packed_names):
        if name in full_names:
            mapping[i] = full_names.index(name)
        else:
            unmatched.append(name)
    if unmatched:
        print(f"  Warning: {len(unmatched)} packed bodies not found "
              f"in full list: {unmatched}")
    return mapping


def remap_body_array(arr, mapping, num_full_bodies):
    T = arr.shape[0]
    last_dim = arr.shape[-1]
    out = np.zeros((T, num_full_bodies, last_dim), dtype=arr.dtype)
    for src, dst in mapping.items():
        out[:, dst, :] = arr[:, src, :]
    return out


def find_body_arrays(data, num_packed_bodies):
    """找出所有 3D 且 shape[1] == num_packed_bodies 的数组（即 body 相关数据）。"""
    keys = []
    for key, arr in data.items():
        if arr.ndim == 3 and arr.shape[1] == num_packed_bodies:
            keys.append(key)
    return keys


def load_body_list(path):
    """从文本文件加载 body 列表（每行一个名字，忽略空行和 # 注释）。"""
    names = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)
    return names


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("onnx_path", help="Path to packed ONNX model")
    parser.add_argument("-o", "--output",
                        help="Output npz path (default: <onnx_dir>/motion.npz)")
    parser.add_argument("--full-bodies",
                        help="Full body list for remap: text file (one per line) "
                             "or comma-separated string")
    parser.add_argument("--no-remap", action="store_true",
                        help="Skip body remapping, output packed body order as-is")
    parser.add_argument("--fps", type=float, default=None,
                        help="Override motion fps (default: read from metadata)")
    parser.add_argument("--min-frames", type=int, default=10,
                        help="Minimum frames to consider an initializer as motion data "
                             "(default: 10)")
    parser.add_argument("--list-initializers", action="store_true",
                        help="Only list discovered initializers, do not extract")
    args = parser.parse_args()

    model = onnx.load(args.onnx_path)

    print(f"Model: {args.onnx_path}")
    print("Metadata:")
    if model.metadata_props:
        for p in model.metadata_props:
            val = p.value if len(p.value) < 200 else p.value[:200] + "..."
            print(f"  {p.key}: {val}")
    else:
        print("  (none)")

    discovered = discover_motion_initializers(model, min_frames=args.min_frames)
    print(f"\nDiscovered {len(discovered)} motion initializers:")
    for name, arr in discovered.items():
        print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")

    if args.list_initializers:
        return

    if not discovered:
        print("Error: no motion initializers found. "
              "Try --min-frames 1 or --list-initializers to inspect.",
              file=sys.stderr)
        sys.exit(1)

    data = dict(discovered)

    packed_bodies_str = get_metadata(model, "body_names")
    packed_bodies = packed_bodies_str.split(",") if packed_bodies_str else None

    full_bodies = None
    if args.full_bodies and not args.no_remap:
        if os.path.isfile(args.full_bodies):
            full_bodies = load_body_list(args.full_bodies)
        else:
            full_bodies = [s.strip() for s in args.full_bodies.split(",")]

    if not args.no_remap and packed_bodies and full_bodies:
        if len(full_bodies) > len(packed_bodies):
            mapping = build_body_mapping(packed_bodies, full_bodies)
            body_keys = find_body_arrays(data, len(packed_bodies))
            print(f"\nRemapping {len(packed_bodies)} -> {len(full_bodies)} bodies "
                  f"({len(mapping)} matched, {len(body_keys)} arrays)")
            for key in body_keys:
                data[key] = remap_body_array(data[key], mapping, len(full_bodies))
                print(f"  {key}: remapped to {data[key].shape}")
        else:
            print(f"\nSkip remap: full list ({len(full_bodies)}) "
                  f"<= packed list ({len(packed_bodies)})")
    elif not args.no_remap and packed_bodies and not full_bodies:
        print(f"\nNote: packed model has {len(packed_bodies)} bodies, "
              f"no --full-bodies provided, output uses packed order as-is. "
              f"If your MotionTrackingHelper expects a different body count, "
              f"provide --full-bodies to remap.")

    if args.fps is not None:
        fps = args.fps
    else:
        fps_meta = get_metadata(model, "motion_fps")
        if fps_meta:
            fps = float(fps_meta)
        else:
            fps = 50.0
            print(f"\nWarning: no motion_fps in metadata, using default {fps}. "
                  f"Override with --fps if incorrect.")
    data["fps"] = np.array([fps])

    if not args.output:
        args.output = os.path.join(os.path.dirname(args.onnx_path) or ".", "motion.npz")

    np.savez(args.output, **data)
    ref_key = next((k for k in ["joint_pos", "joint_vel"] if k in data), None)
    if ref_key:
        frames = data[ref_key].shape[0]
        print(f"\nSaved to {args.output}")
        print(f"  frames={frames}, fps={fps}, duration={frames / fps:.1f}s")
    else:
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
