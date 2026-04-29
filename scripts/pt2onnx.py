# Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch 模型转 ONNX 通用脚本
- 自动检测模型类型（MLP / LSTM / GRU）
- 自动推断输入输出维度
- LSTM模型自动进行状态外置化
"""

import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import argparse
import sys

class ModelAnalyzer:
    """模型分析器：自动检测模型类型和参数"""

    def __init__(self, model):
        self.model = model
        self.model_type = None  # 'mlp', 'lstm', 'gru'
        self.input_dim = None
        self.output_dim = None
        self.hidden_size = None
        self.num_layers = None
        self.has_internal_state = False
        self.num_inputs = 1  # 输入数量
        self.input_names = []  # 输入名称列表
        self.input_dims = []  # 输入维度列表

    def analyze(self):
        """全自动分析模型"""
        print("\n[模型分析] 正在检测模型类型和参数...")

        # 1. 检测输入数量和名称
        self._detect_inputs()

        # 2. 检测是否有LSTM/GRU状态
        self._detect_rnn_type()

        # 3. 推断输入输出维度
        self._infer_io_dims()

        # 4. 打印分析结果
        self._print_summary()

        return self

    def _detect_inputs(self):
        """检测模型输入数量和名称"""
        try:
            if hasattr(self.model, 'graph'):
                # TorchScript模型，从graph获取输入信息
                inputs = list(self.model.graph.inputs())
                # 第一个通常是self，跳过
                actual_inputs = []
                for inp in inputs:
                    inp_str = str(inp)
                    # 跳过self参数（检查是否以self.开头）
                    if inp_str.strip().startswith('self.'):
                        continue
                    actual_inputs.append(inp)

                self.num_inputs = len(actual_inputs)

                # 提取输入名称
                for inp in actual_inputs:
                    inp_str = str(inp)
                    # 解析类似 "obs.1 defined in ..." 的字符串
                    # 提取第一个单词作为变量名
                    name = inp_str.split()[0].split('.')[0]
                    self.input_names.append(name)

                if self.num_inputs > 1:
                    print(f"   检测到多输入模型: {self.num_inputs}个输入")
                    print(f"   输入名称: {self.input_names}")
                elif self.num_inputs == 0:
                    # 可能检测失败，使用默认
                    print("   ⚠️  输入检测异常，使用默认单输入")
                    self.num_inputs = 1
                    self.input_names = ['obs']
        except Exception as e:
            print(f"   输入检测失败，使用默认单输入: {e}")
            self.num_inputs = 1
            self.input_names = ['obs']

    def _detect_rnn_type(self):
        """检测RNN类型"""
        if hasattr(self.model, 'hidden_state') and hasattr(self.model, 'cell_state'):
            self.model_type = 'lstm'
            self.has_internal_state = True
            self.num_layers, _, self.hidden_size = self.model.hidden_state.shape
            print("   ✅ 检测到 LSTM 模型")
            print(f"      hidden_size: {self.hidden_size}")
            print(f"      num_layers: {self.num_layers}")
        elif hasattr(self.model, 'hidden_state'):
            self.model_type = 'gru'
            self.has_internal_state = True
            self.num_layers, _, self.hidden_size = self.model.hidden_state.shape
            print("   ✅ 检测到 GRU 模型")
            print(f"      hidden_size: {self.hidden_size}")
            print(f"      num_layers: {self.num_layers}")
        else:
            self.model_type = 'mlp'
            self.has_internal_state = False
            print("   ✅ 检测到 MLP 模型（无状态）")

    def _infer_io_dims(self):
        """推断输入输出维度"""
        print("   正在推断输入输出维度...")

        # 尝试常见维度组合
        if self.num_inputs == 1:
            common_dims = [750, 512, 256, 128, 64, 47, 45, 39, 37, 1024, 2048, 4373]

            for dim in common_dims:
                try:
                    dummy_input = torch.randn(1, dim, dtype=torch.float32)

                    # 如果有状态，需要重置
                    if self.has_internal_state and hasattr(self.model, 'reset_memory'):
                        self.model.reset_memory()

                    with torch.no_grad():
                        output = self.model(dummy_input)

                    self.input_dim = dim
                    self.input_dims = [dim]
                    self.output_dim = output.shape[-1]
                    print(f"   ✅ 推断成功: 输入={self.input_dim}, 输出={self.output_dim}")
                    return

                except Exception:
                    continue

        else:
            # 多输入模型，尝试不同的维度组合
            print(f"   尝试推断{self.num_inputs}个输入的维度...")

            # 常见的多输入维度组合
            # 支持2D和3D输入
            multi_input_configs = [
                # 2D输入: (obs_dim, hist_dim)
                ([39, 390], [39, 39*10]),
                ([195, 1950], [195, 195*10]),
                ([47, 470], [47, 47*10]),
                ([64, 640], [64, 64*10]),
                ([128, 1280], [128, 128*10]),
                # 3D输入: obs是2D, hist是3D [batch, seq, dim]
                # 用tuple表示3D: (batch, seq, dim)
                ([(1, 39), (1, 10, 39)], [(1, 39), (1, 5, 39)]),
                ([(1, 47), (1, 10, 47)], [(1, 47), (1, 5, 47)]),
                ([(1, 195), (1, 10, 195)], [(1, 195), (1, 5, 195)]),
            ]

            for config in multi_input_configs:
                for dims in config:
                    if len(dims) != self.num_inputs:
                        continue

                    try:
                        dummy_inputs = []
                        for dim in dims:
                            if isinstance(dim, tuple):
                                # 3D输入
                                dummy_inputs.append(torch.randn(*dim, dtype=torch.float32))
                            else:
                                # 2D输入
                                dummy_inputs.append(torch.randn(1, dim, dtype=torch.float32))

                        if self.has_internal_state and hasattr(self.model, 'reset_memory'):
                            self.model.reset_memory()

                        with torch.no_grad():
                            output = self.model(*dummy_inputs)

                        # 记录维度（对于3D输入，记录最后一维）
                        self.input_dims = []
                        for dim in dims:
                            if isinstance(dim, tuple):
                                self.input_dims.append(dim)  # 保存完整shape
                            else:
                                self.input_dims.append(dim)

                        first = self.input_dims[0]
                        self.input_dim = first if not isinstance(first, tuple) else first[-1]
                        self.output_dim = output.shape[-1]
                        print(f"   ✅ 推断成功: 输入={self.input_dims}, 输出={self.output_dim}")
                        return

                    except Exception:
                        continue

        # 如果常见维度都失败，尝试从模型结构推断
        print("   ⚠️  常见维度检测失败，尝试从模型结构推断...")

        # 对于LSTM模型，尝试从LSTM层推断
        if self.model_type == 'lstm' and hasattr(self.model, 'memory'):
            try:
                lstm_layer = self.model.memory
                if hasattr(lstm_layer, 'input_size'):
                    self.input_dim = lstm_layer.input_size
                    self.input_dims = [self.input_dim]
                    print(f"   从LSTM层推断输入维度: {self.input_dim}")

                    dummy_input = torch.randn(1, self.input_dim, dtype=torch.float32)
                    if hasattr(self.model, 'reset_memory'):
                        self.model.reset_memory()
                    with torch.no_grad():
                        output = self.model(dummy_input)
                    self.output_dim = output.shape[-1]
                    print(f"   ✅ 推断成功: 输入={self.input_dim}, 输出={self.output_dim}")
                    return
            except Exception as e:
                print(f"   LSTM推断失败: {e}")

        # 尝试从第一层参数推断
        try:
            if hasattr(self.model, 'parameters'):
                first_param = next(self.model.parameters())
                if len(first_param.shape) >= 2:
                    inferred_dim = first_param.shape[1]
                    print(f"   从第一层参数推断输入维度: {inferred_dim}")

                    # 对于多输入模型，尝试不同的分配策略
                    if self.num_inputs > 1:
                        # 策略1: 平均分配
                        avg_dim = inferred_dim // self.num_inputs
                        dims = [avg_dim] * self.num_inputs

                        # 策略2: 第一个是主输入，其余是历史
                        dims_alt = [inferred_dim // (self.num_inputs + 9)] * self.num_inputs
                        dims_alt[1] = inferred_dim - dims_alt[0]

                        for dims in [dims, dims_alt]:
                            try:
                                dummy_inputs = [torch.randn(1, dim, dtype=torch.float32) for dim in dims]
                                if self.has_internal_state and hasattr(self.model, 'reset_memory'):
                                    self.model.reset_memory()
                                with torch.no_grad():
                                    output = self.model(*dummy_inputs)

                                self.input_dims = dims
                                self.input_dim = dims[0]
                                self.output_dim = output.shape[-1]
                                print(f"   ✅ 推断成功: 输入={self.input_dims}, 输出={self.output_dim}")
                                return
                            except Exception:
                                continue
                    else:
                        # 单输入
                        dummy_input = torch.randn(1, inferred_dim, dtype=torch.float32)
                        if self.has_internal_state and hasattr(self.model, 'reset_memory'):
                            self.model.reset_memory()
                        with torch.no_grad():
                            output = self.model(dummy_input)

                        self.input_dim = inferred_dim
                        self.input_dims = [inferred_dim]
                        self.output_dim = output.shape[-1]
                        print(f"   ✅ 推断成功: 输入={self.input_dim}, 输出={self.output_dim}")
                        return
        except Exception as e:
            print(f"   参数推断失败: {e}")

        raise RuntimeError("❌ 无法自动检测模型输入输出维度，请使用 --input-dims 手动指定")

    def _print_summary(self):
        """打印分析摘要"""
        print("\n   [分析摘要]")
        print(f"   模型类型: {self.model_type.upper()}")
        if self.num_inputs > 1:
            print(f"   输入数量: {self.num_inputs}")
            for i, (name, dim) in enumerate(zip(self.input_names, self.input_dims)):
                print(f"   输入{i+1}: {name}[{dim}]")
        else:
            print(f"   输入维度: {self.input_dim}")
        print(f"   输出维度: {self.output_dim}")
        if self.has_internal_state:
            print(f"   隐藏层大小: {self.hidden_size}")
            print(f"   RNN层数: {self.num_layers}")
            print("   状态管理: 需要外置化")
        else:
            print("   状态管理: 无状态模型")


class LSTMStateExternalizer(torch.nn.Module):
    """LSTM状态外置化包装器"""

    def __init__(self, original_model):
        super().__init__()
        self.model = original_model

    def forward(self, obs, h_in, c_in):
        """
        将外部状态注入模型，执行推理，返回更新后的状态

        Args:
            obs: 观测输入 [batch, obs_dim]
            h_in: 输入hidden state [num_layers, batch, hidden_size]
            c_in: 输入cell state [num_layers, batch, hidden_size]

        Returns:
            action: 动作输出 [batch, action_dim]
            h_out: 输出hidden state [num_layers, batch, hidden_size]
            c_out: 输出cell state [num_layers, batch, hidden_size]
        """
        # 注入状态
        self.model.hidden_state.copy_(h_in)
        self.model.cell_state.copy_(c_in)

        # 推理
        action = self.model(obs)

        # 提取更新后的状态
        h_out = self.model.hidden_state.clone()
        c_out = self.model.cell_state.clone()

        return action, h_out, c_out


class GRUStateExternalizer(torch.nn.Module):
    """GRU状态外置化包装器"""

    def __init__(self, original_model):
        super().__init__()
        self.model = original_model

    def forward(self, obs, h_in):
        self.model.hidden_state.copy_(h_in)
        action = self.model(obs)
        h_out = self.model.hidden_state.clone()
        return action, h_out


def convert_pt_to_onnx_universal(
    policy_path,
    onnx_path,
    input_name="obs",
    output_name="action",
    opset_version=11,
    verify=True
):
    """
    通用PyTorch到ONNX转换器
    - 自动检测模型类型（MLP/LSTM/GRU）
    - 自动推断所有维度
    - LSTM/GRU自动进行状态外置化
    """

    print("=" * 70)
    print("Universal PyTorch to ONNX Converter")
    print("=" * 70)

    # ============================================================
    # 1. 加载模型
    # ============================================================
    print(f"\n[1/7] 加载 PyTorch 模型: {policy_path}")
    try:
        model = torch.jit.load(policy_path, map_location="cpu")
        model.eval()
        # 强制转换到float32（处理FP16模型）
        model = model.float()
        print("✅ 模型加载成功 (TorchScript)")
    except Exception as e1:
        print("   TorchScript加载失败，尝试torch.load...")
        try:
            model = torch.load(policy_path, map_location="cpu", weights_only=False)
            if hasattr(model, 'eval'):
                model.eval()
            if hasattr(model, 'float'):
                model = model.float()
            print("✅ 模型加载成功 (torch.load)")
        except Exception as e2:
            print("❌ 模型加载失败:")
            print(f"   TorchScript错误: {str(e1)[:100]}")
            print(f"   torch.load错误: {str(e2)[:100]}")

            # 提供解决建议
            if "No module named" in str(e2):
                module_name = str(e2).split("'")[1] if "'" in str(e2) else "unknown"
                print("\n💡 解决建议:")
                print(f"   该模型依赖外部模块 '{module_name}'，无法单独加载。")
                print("   请使用以下方法之一:")
                print("   1. 在训练代码目录下运行转换脚本")
                print("   2. 将模型重新导出为TorchScript格式（推荐）:")
                print("      traced = torch.jit.trace(model, example_input)")
                print("      traced.save('model_jit.pt')")
                print("   3. 使用 model_jittv2.pt（如果已有TorchScript版本）")
            elif "constants.pkl" in str(e1):
                print("\n💡 解决建议:")
                print("   该文件不是有效的TorchScript模型。")
                print("   请确认:")
                print("   1. 文件是否完整（未损坏）")
                print("   2. 是否有对应的TorchScript版本（如 model_jit.pt）")
                print("   3. 或在训练环境中重新导出为TorchScript格式")

            sys.exit(1)

    # ============================================================
    # 2. 分析模型
    # ============================================================
    print("\n[2/7] 分析模型结构...")
    try:
        analyzer = ModelAnalyzer(model).analyze()
    except Exception as e:
        print(f"❌ 模型分析失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ============================================================
    # 3. 准备转换
    # ============================================================
    print("\n[3/7] 准备转换...")

    if analyzer.model_type == 'lstm':
        print("   策略: LSTM状态外置化")
        wrapper = LSTMStateExternalizer(model)
        wrapper.eval()

        if analyzer.num_inputs > 1:
            dummy_obs_list = [torch.randn(1, dim, dtype=torch.float32) for dim in analyzer.input_dims]
            dummy_h = torch.zeros(analyzer.num_layers, 1, analyzer.hidden_size, dtype=torch.float32)
            dummy_c = torch.zeros(analyzer.num_layers, 1, analyzer.hidden_size, dtype=torch.float32)
            dummy_inputs = tuple(dummy_obs_list + [dummy_h, dummy_c])

            input_names = analyzer.input_names + ["h_in", "c_in"]
            output_names = ["action", "h_out", "c_out"]
            dynamic_axes = {}
            for name in analyzer.input_names:
                dynamic_axes[name] = {0: "batch_size"}
            dynamic_axes.update({
                "h_in": {1: "batch_size"},
                "c_in": {1: "batch_size"},
                "action": {0: "batch_size"},
                "h_out": {1: "batch_size"},
                "c_out": {1: "batch_size"}
            })
        else:
            dummy_obs = torch.randn(1, analyzer.input_dim, dtype=torch.float32)
            dummy_h = torch.zeros(analyzer.num_layers, 1, analyzer.hidden_size, dtype=torch.float32)
            dummy_c = torch.zeros(analyzer.num_layers, 1, analyzer.hidden_size, dtype=torch.float32)
            dummy_inputs = (dummy_obs, dummy_h, dummy_c)

            input_names = ["obs", "h_in", "c_in"]
            output_names = ["action", "h_out", "c_out"]
            dynamic_axes = {
                "obs": {0: "batch_size"},
                "h_in": {1: "batch_size"},
                "c_in": {1: "batch_size"},
                "action": {0: "batch_size"},
                "h_out": {1: "batch_size"},
                "c_out": {1: "batch_size"}
            }

    elif analyzer.model_type == 'gru':
        print("   策略: GRU状态外置化")
        wrapper = GRUStateExternalizer(model)
        wrapper.eval()

        if analyzer.num_inputs > 1:
            dummy_obs_list = [torch.randn(1, dim, dtype=torch.float32) for dim in analyzer.input_dims]
            dummy_h = torch.zeros(analyzer.num_layers, 1, analyzer.hidden_size, dtype=torch.float32)
            dummy_inputs = tuple(dummy_obs_list + [dummy_h])

            input_names = analyzer.input_names + ["h_in"]
            output_names = ["action", "h_out"]
            dynamic_axes = {}
            for name in analyzer.input_names:
                dynamic_axes[name] = {0: "batch_size"}
            dynamic_axes.update({
                "h_in": {1: "batch_size"},
                "action": {0: "batch_size"},
                "h_out": {1: "batch_size"}
            })
        else:
            dummy_obs = torch.randn(1, analyzer.input_dim, dtype=torch.float32)
            dummy_h = torch.zeros(analyzer.num_layers, 1, analyzer.hidden_size, dtype=torch.float32)
            dummy_inputs = (dummy_obs, dummy_h)

            input_names = ["obs", "h_in"]
            output_names = ["action", "h_out"]
            dynamic_axes = {
                "obs": {0: "batch_size"},
                "h_in": {1: "batch_size"},
                "action": {0: "batch_size"},
                "h_out": {1: "batch_size"}
            }

    else:  # MLP
        print("   策略: 直接转换（无状态）")
        wrapper = model

        if analyzer.num_inputs > 1:
            dummy_obs_list = []
            for dim in analyzer.input_dims:
                if isinstance(dim, tuple):
                    # 3D输入
                    dummy_obs_list.append(torch.randn(*dim, dtype=torch.float32))
                else:
                    # 2D输入
                    dummy_obs_list.append(torch.randn(1, dim, dtype=torch.float32))
            dummy_inputs = tuple(dummy_obs_list)

            input_names = analyzer.input_names
            output_names = [output_name]
            dynamic_axes = {}
            for i, (name, dim) in enumerate(zip(analyzer.input_names, analyzer.input_dims)):
                if isinstance(dim, tuple):
                    # 3D输入: 第0维和第1维都是dynamic
                    dynamic_axes[name] = {0: "batch_size", 1: "seq_len"}
                else:
                    # 2D输入
                    dynamic_axes[name] = {0: "batch_size"}
            dynamic_axes[output_name] = {0: "batch_size"}
        else:
            dummy_obs = torch.randn(1, analyzer.input_dim, dtype=torch.float32)
            dummy_inputs = dummy_obs

            input_names = [input_name]
            output_names = [output_name]
            dynamic_axes = {
                input_name: {0: "batch_size"},
                output_name: {0: "batch_size"}
            }

    print("   ✅ 准备完成")

    # ============================================================
    # 4. 测试推理
    # ============================================================
    print("\n[4/7] 测试 PyTorch 推理...")
    try:
        if analyzer.has_internal_state and hasattr(model, 'reset_memory'):
            model.reset_memory()

        with torch.no_grad():
            if analyzer.model_type == 'lstm':
                if isinstance(dummy_inputs, tuple):
                    action, h_out, c_out = wrapper(*dummy_inputs)
                else:
                    action, h_out, c_out = wrapper(dummy_inputs)
                print("✅ 推理成功")
                print(f"   action: {action.shape}, 范围: [{action.min():.4f}, {action.max():.4f}]")
                print(f"   h_out: {h_out.shape}")
                print(f"   c_out: {c_out.shape}")
            elif analyzer.model_type == 'gru':
                if isinstance(dummy_inputs, tuple):
                    action, h_out = wrapper(*dummy_inputs)
                else:
                    action, h_out = wrapper(dummy_inputs)
                print("✅ 推理成功")
                print(f"   action: {action.shape}, 范围: [{action.min():.4f}, {action.max():.4f}]")
                print(f"   h_out: {h_out.shape}")
            else:
                if isinstance(dummy_inputs, tuple):
                    action = wrapper(*dummy_inputs)
                else:
                    action = wrapper(dummy_inputs)
                print("✅ 推理成功")
                print(f"   action: {action.shape}, 范围: [{action.min():.4f}, {action.max():.4f}]")
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ============================================================
    # 5. 导出 ONNX
    # ============================================================
    print(f"\n[5/7] 导出 ONNX 模型到 {onnx_path}...")
    try:
        # 对于TorchScript模型，需要script包装器
        if analyzer.has_internal_state:
            print("   使用 torch.jit.script 模式...")
            wrapper = torch.jit.script(wrapper)

        torch.onnx.export(
            wrapper,
            dummy_inputs,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            export_params=True,
            do_constant_folding=True,
            verbose=False
        )
        print("✅ ONNX 模型导出成功")
    except Exception as e:
        print(f"❌ ONNX 导出失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ============================================================
    # 6. 验证 ONNX
    # ============================================================
    print("\n[6/7] 验证 ONNX 模型...")
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX 模型结构检查通过")

        print("\n   ONNX 输入:")
        for inp in onnx_model.graph.input:
            dims = [d.dim_value if d.dim_value > 0 else "dynamic" for d in inp.type.tensor_type.shape.dim]
            print(f"   - {inp.name}: {dims}")

        print("\n   ONNX 输出:")
        for out in onnx_model.graph.output:
            dims = [d.dim_value if d.dim_value > 0 else "dynamic" for d in out.type.tensor_type.shape.dim]
            print(f"   - {out.name}: {dims}")
    except Exception as e:
        print(f"❌ ONNX 验证失败: {e}")
        sys.exit(1)

    # ============================================================
    # 7. 对比验证
    # ============================================================
    if verify:
        print("\n[7/7] 对比 PyTorch vs ONNX 输出...")
        try:
            # 重置状态
            if analyzer.has_internal_state and hasattr(model, 'reset_memory'):
                model.reset_memory()

            # PyTorch推理
            with torch.no_grad():
                if analyzer.model_type == 'lstm':
                    if isinstance(dummy_inputs, tuple):
                        pt_action, pt_h, pt_c = wrapper(*dummy_inputs)
                    else:
                        pt_action, pt_h, pt_c = wrapper(dummy_inputs)
                elif analyzer.model_type == 'gru':
                    if isinstance(dummy_inputs, tuple):
                        pt_action, pt_h = wrapper(*dummy_inputs)
                    else:
                        pt_action, pt_h = wrapper(dummy_inputs)
                else:
                    if isinstance(dummy_inputs, tuple):
                        pt_action = wrapper(*dummy_inputs)
                    else:
                        pt_action = wrapper(dummy_inputs)

            # ONNX推理
            ort_session = ort.InferenceSession(onnx_path)

            if analyzer.model_type == 'lstm':
                ort_inputs = {
                    "obs": dummy_obs.numpy(),
                    "h_in": dummy_h.numpy(),
                    "c_in": dummy_c.numpy()
                }
                ort_outputs = ort_session.run(None, ort_inputs)
                onnx_action, onnx_h, onnx_c = ort_outputs

                action_diff = np.abs(pt_action.numpy() - onnx_action)
                h_diff = np.abs(pt_h.numpy() - onnx_h)
                c_diff = np.abs(pt_c.numpy() - onnx_c)
                max_diff = max(action_diff.max(), h_diff.max(), c_diff.max())

                print(f"   action 最大差异: {action_diff.max():.6f}")
                print(f"   h_out 最大差异: {h_diff.max():.6f}")
                print(f"   c_out 最大差异: {c_diff.max():.6f}")

            elif analyzer.model_type == 'gru':
                ort_inputs = {
                    "obs": dummy_obs.numpy(),
                    "h_in": dummy_h.numpy()
                }
                ort_outputs = ort_session.run(None, ort_inputs)
                onnx_action, onnx_h = ort_outputs

                action_diff = np.abs(pt_action.numpy() - onnx_action)
                h_diff = np.abs(pt_h.numpy() - onnx_h)
                max_diff = max(action_diff.max(), h_diff.max())

                print(f"   action 最大差异: {action_diff.max():.6f}")
                print(f"   h_out 最大差异: {h_diff.max():.6f}")

            else:
                # 准备ONNX输入
                if analyzer.num_inputs > 1:
                    ort_inputs = {}
                    for i, (name, dummy_input) in enumerate(zip(analyzer.input_names, dummy_inputs)):
                        ort_inputs[name] = dummy_input.numpy()
                else:
                    ort_inputs = {input_name: dummy_inputs.numpy()}

                onnx_action = ort_session.run(None, ort_inputs)[0]

                action_diff = np.abs(pt_action.numpy() - onnx_action)
                max_diff = action_diff.max()

                print(f"   action 最大差异: {max_diff:.6f}")
                print(f"   action 平均差异: {action_diff.mean():.6f}")

            # 判断精度
            if max_diff < 1e-5:
                print("   ✅ 转换完美！差异 < 1e-5")
            elif max_diff < 1e-3:
                print("   ✅ 转换成功！差异 < 1e-3")
            else:
                print(f"   ⚠️  差异较大: {max_diff:.6f}")

            # 性能测试
            print("\n   [性能测试] 推理速度对比...")
            import time
            num_runs = 100

            # PyTorch
            if analyzer.has_internal_state and hasattr(model, 'reset_memory'):
                model.reset_memory()
            start = time.time()
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = wrapper(*dummy_inputs) if isinstance(dummy_inputs, tuple) else wrapper(dummy_inputs)
            pt_time = (time.time() - start) / num_runs * 1000

            # ONNX
            start = time.time()
            for _ in range(num_runs):
                _ = ort_session.run(None, ort_inputs)
            onnx_time = (time.time() - start) / num_runs * 1000

            print(f"   PyTorch: {pt_time:.2f} ms")
            print(f"   ONNX: {onnx_time:.2f} ms")
            print(f"   加速比: {pt_time/onnx_time:.2f}x")

        except Exception as e:
            print(f"   ⚠️  验证失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[7/7] 跳过验证")

    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "=" * 70)
    print("🎉 转换完成！")
    print(f"   输入文件: {policy_path}")
    print(f"   输出文件: {onnx_path}")
    print(f"   模型类型: {analyzer.model_type.upper()}")
    print(f"   输入维度: {analyzer.input_dim}")
    print(f"   输出维度: {analyzer.output_dim}")

    if analyzer.has_internal_state:
        print(f"   隐藏层大小: {analyzer.hidden_size}")
        print(f"   RNN层数: {analyzer.num_layers}")
        print("\n   使用方式:")
        if analyzer.model_type == 'lstm':
            print("   1. 初始化: h=0, c=0")
            print("   2. 推理: action, h, c = model(obs, h, c)")
            print("   3. 下一帧: 使用上一帧的 h, c 作为输入")
        else:
            print("   1. 初始化: h=0")
            print("   2. 推理: action, h = model(obs, h)")
            print("   3. 下一帧: 使用上一帧的 h 作为输入")
    else:
        print("\n   使用方式:")
        print("   action = model(obs)  # 无状态，直接推理")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="通用 PyTorch 模型转 ONNX（自动检测模型类型）",
        epilog="""
示例:
  # 自动检测并转换（推荐）
  python pt2onnx.py -i model.pt -o model.onnx

  # MLP模型
  python pt2onnx.py -i walk.pt -o walk.onnx

  # LSTM模型（自动进行状态外置化）
  python pt2onnx.py -i motion.pt -o motion.onnx

  # 自定义输入输出名称
  python pt2onnx.py -i model.pt -o model.onnx --input-name observations --output-name actions
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入 PyTorch 模型路径"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="输出 ONNX 模型路径"
    )
    parser.add_argument(
        "--input-name",
        type=str,
        default="obs",
        help="ONNX 输入节点名称 (默认: obs, 仅用于MLP模型)"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="action",
        help="ONNX 输出节点名称 (默认: action, 仅用于MLP模型)"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset 版本 (默认: 11)"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="跳过验证步骤"
    )

    args = parser.parse_args()

    convert_pt_to_onnx_universal(
        policy_path=args.input,
        onnx_path=args.output,
        input_name=args.input_name,
        output_name=args.output_name,
        opset_version=args.opset,
        verify=not args.no_verify
    )


if __name__ == "__main__":
    main()
