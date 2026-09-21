#!/usr/bin/env python3
"""
BitPartitionedGroupMoE - 按 bit 分区的 grouped_gemm MoE 实现

设计思路 ：
- 只保留按 bit 分开的权重，内存最优
- 优化前向计算：对每个 token，先收集所有需要的 expert-bit 组合，批量计算
- 目标：速度接近完整合并权重版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import time
import gc


class BitPartitionedGroupMoE(nn.Module):
    """
    按 bit 分区的 grouped_gemm MoE

    存储结构（无冗余）：
        - gate: 复用原始 router
        - bit_weights.gate_up: ParameterDict[str, Tensor] - 按 bit 分开的权重
        - bit_weights.down: ParameterDict[str, Tensor] - 按 bit 分开的权重
    """

    def __init__(
        self,
        gate,
        num_experts,
        hidden_size,
        intermediate_size,
        top_k=6,
        shared_expert=None,
        shared_expert_gate=None,
        shared_experts=None,
    ):
        super().__init__()

        self.gate = gate
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.shared_expert = shared_expert
        self.shared_expert_gate = shared_expert_gate
        self.shared_experts = shared_experts

        # 这些会在后面填充
        self.bit_list = []

        # 创建 bit_weights 子模块，存放按 bit 分开的权重
        class BitWeights(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_up = nn.ParameterDict()
                self.down = nn.ParameterDict()

        self.bit_weights = BitWeights()
        # 紧凑存储格式：
        # - gate_up[bit_str]: (total_neurons_2x, H)  # 所有 expert 的 gate/up 权重拼接，2x 因为 gate+up
        # - down[bit_str]: (H, total_neurons)       # 所有 expert 的 down 权重拼接
        # - expert_offsets[bit_str]: (E+1,)          # 每个 expert 的 start idx，cumsum 格式
        self.inter_size_by_bit = {}
        self.expert_offsets = {}  # bit_str -> LongTensor: expert_idx -> start_pos (cumsum格式)

    @classmethod
    def from_build_block(cls, build_block, layer_metadata):
        """
        从 MoEBuildBlock 重构为 BitPartitionedGroupMoE

        参数：
            build_block: MoEBuildBlock 实例（包含 DartMoQHybridWrapper）
            layer_metadata: 量化过程中保存的元数据

        返回：
            BitPartitionedGroupMoE 实例
        """
        # 从 meta 中获取信息
        num_experts = layer_metadata['num_experts']
        hidden_size = layer_metadata['hidden_size']
        intermediate_size = layer_metadata['intermediate_size']
        bit_list = layer_metadata['bit_list']
        expert_bit_indices = layer_metadata['expert_bit_indices']

        # 创建实例
        moe = cls(
            gate=build_block.gate,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            top_k=getattr(build_block, 'top_k', None) or layer_metadata.get('top_k', 6),
            shared_expert=getattr(build_block, 'shared_expert', None),
            shared_expert_gate=getattr(build_block, 'shared_expert_gate', None),
            shared_experts=getattr(build_block, 'shared_experts', None),
        )

        moe.bit_list = bit_list

        # 获取 dtype 和 device
        dtype = next(build_block.parameters()).dtype
        device = next(build_block.parameters()).device

        # 第一步：收集每个 expert 在每个 bit 的神经元数
        neurons_by_bit_expert = defaultdict(list)  # bit -> list[int]
        for expert_idx in range(num_experts):
            bit_indices = expert_bit_indices[expert_idx]
            for bit in bit_list:
                indices = bit_indices.get(bit, [])
                neurons_by_bit_expert[bit].append(len(indices))

        # 第二步：为每个 bit 初始化紧凑格式的权重张量（无 padding）
        for bit in bit_list:
            bit_str = str(bit)
            neuron_counts = neurons_by_bit_expert[bit]
            total_neurons = sum(neuron_counts)
            if total_neurons == 0:
                continue

            # 计算 expert offsets (cumsum 格式，方便索引)
            expert_offsets = torch.zeros(num_experts + 1, dtype=torch.long, device=device)
            expert_offsets[1:] = torch.tensor(neuron_counts, dtype=torch.long, device=device).cumsum(dim=0)

            # (total_neurons_2x, H): 所有 expert 的 gate/up 拼接，每个 expert 是 [gate, up]
            gate_up = torch.zeros(2 * total_neurons, hidden_size, dtype=dtype, device=device)
            # (H, total_neurons): 所有 expert 的 down 拼接
            down = torch.zeros(hidden_size, total_neurons, dtype=dtype, device=device)

            moe.bit_weights.gate_up[bit_str] = nn.Parameter(gate_up, requires_grad=False)
            moe.bit_weights.down[bit_str] = nn.Parameter(down, requires_grad=False)
            moe.expert_offsets[bit_str] = expert_offsets
            moe.inter_size_by_bit[bit] = total_neurons  # 这里存 total 只是标记该 bit 有神经元

        # 第三步：从每个 expert 的 sub_expert 中提取权重，填充到紧凑格式中
        for expert_idx in range(num_experts):
            wrapper = build_block.experts[expert_idx]  # DartMoQHybridWrapper

            # 先为每个 bit 建立 sub_expert 的映射
            bit_to_subexpert = {}
            for sub_expert in wrapper.sub_experts:
                bit = sub_expert._quant_bit
                bit_to_subexpert[bit] = sub_expert

            # 按 bit_list 中的 bit 顺序处理
            for bit in bit_list:
                bit_str = str(bit)
                if bit_str not in moe.bit_weights.gate_up:
                    continue

                if bit not in bit_to_subexpert:
                    continue

                sub_expert = bit_to_subexpert[bit]
                n_neurons = sub_expert.gate_proj.weight.shape[0]
                if n_neurons == 0:
                    continue

                # 获取该 expert 在该 bit 中的位置
                start = moe.expert_offsets[bit_str][expert_idx]
                end = moe.expert_offsets[bit_str][expert_idx + 1]

                # 填充 gate 和 up 到紧凑格式: [gate1, up1, gate2, up2, ...]
                moe.bit_weights.gate_up[bit_str][2*start : 2*start + n_neurons] = sub_expert.gate_proj.weight.data
                moe.bit_weights.gate_up[bit_str][2*start + n_neurons : 2*end] = sub_expert.up_proj.weight.data

                # 填充 down 到紧凑格式
                moe.bit_weights.down[bit_str][:, start:end] = sub_expert.down_proj.weight.data

                # 立即清理这个 sub_expert 的权重，释放内存
                del sub_expert.gate_proj
                del sub_expert.up_proj
                del sub_expert.down_proj
                del bit_to_subexpert[bit]

            # 清理这个 wrapper 的引用
            del bit_to_subexpert
            if hasattr(wrapper, 'sub_experts'):
                del wrapper.sub_experts
            if hasattr(wrapper, 'bit_to_indices'):
                del wrapper.bit_to_indices

            # 清理 build_block.experts 中这个位置的引用
            build_block.experts[expert_idx] = None

        # 清理 build_block 的 experts 列表引用
        if hasattr(build_block, 'experts'):
            del build_block.experts

        # 强制垃圾回收，释放 CPU 内存
        gc.collect()

        return moe

    def forward(self, hidden_states):
        t0 = time.time()

        batch_size, seq_len, hidden_dim = hidden_states.shape
        x = hidden_states.reshape(-1, hidden_dim)

        final_hidden_states = torch.zeros_like(x)
        t1 = time.time()

        # Shared expert
        if self.shared_expert is not None and self.shared_expert_gate is not None:
            shared_out = self.shared_expert(x)
            shared_gate_val = torch.sigmoid(self.shared_expert_gate(x))
            final_hidden_states.add_(shared_out * shared_gate_val)
            # Cleanup shared expert variables
            del shared_out, shared_gate_val
        t2 = time.time()

        # Router
        gate_output = self.gate(x)
        if isinstance(gate_output, tuple):
            _, topk_weights, topk_indices = gate_output
        else:
            router_logits = gate_output.softmax(dim=-1)
            topk_weights, topk_indices = router_logits.topk(self.top_k, dim=-1)
            del router_logits
        # Cleanup gate_output
        del gate_output
        t3 = time.time()

        # 详细统计 compute 内部各部分时间
        t_mask_total = 0.0
        t_gate_up_matmul_total = 0.0
        t_silu_total = 0.0
        t_down_matmul_total = 0.0
        t_accum_total = 0.0

        active_experts_total = 0

        # 优化结构：expert -> bit (GPU 反向索引，参考 deepseek)
        t_mask_start = time.time()

        # Flatten: (N, top_k) -> (N * top_k,)
        flat_expert_indices = topk_indices.flatten()
        flat_expert_weights = topk_weights.flatten()
        flat_token_indices = torch.arange(x.shape[0], device=x.device).repeat_interleave(self.top_k)

        # 按 expert 排序
        idxs = flat_expert_indices.argsort()
        sorted_experts = flat_expert_indices[idxs]
        sorted_weights = flat_expert_weights[idxs]
        sorted_tokens = flat_token_indices[idxs]

        # 统计每个 expert 的 token 数，得到结束位置
        tokens_per_expert = sorted_experts.bincount(minlength=self.num_experts).cpu().numpy().cumsum(0)

        t_mask_end = time.time()
        t_mask_total += t_mask_end - t_mask_start

        # 对每个 expert 处理所有选择它的 token
        for expert_idx in range(self.num_experts):
            end_idx = tokens_per_expert[expert_idx]
            start_idx = 0 if expert_idx == 0 else tokens_per_expert[expert_idx - 1]

            if start_idx == end_idx:
                continue

            active_experts_total += 1

            # 取当前 expert 的所有 token
            exp_token_idx = sorted_tokens[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_weights = sorted_weights[start_idx:end_idx].unsqueeze(1)

            # 对同一个 expert 的所有 token，串行处理所有 bit
            expert_out = torch.zeros_like(expert_tokens)

            for bit in self.bit_list:
                bit_str = str(bit)
                if bit_str not in self.bit_weights.gate_up:
                    continue

                gate_up = self.bit_weights.gate_up[bit_str]
                down = self.bit_weights.down[bit_str]
                offsets = self.expert_offsets[bit_str]
                start = offsets[expert_idx]
                end = offsets[expert_idx + 1]
                actual_inter_size = end - start

                if actual_inter_size == 0:
                    continue

                # 获取这个 expert 的权重（紧凑格式）
                e_gate_up = gate_up[2*start : 2*end]  # (2*actual_I_b, H)
                e_down = down[:, start:end]           # (H, actual_I_b)

                t_gate_up_start = time.time()
                gate_up_out = expert_tokens @ e_gate_up.t()
                t_gate_up_end = time.time()
                t_gate_up_matmul_total += t_gate_up_end - t_gate_up_start

                gate_out = gate_up_out[:, :actual_inter_size]
                up_out = gate_up_out[:, actual_inter_size:]
                del gate_up_out

                # SILU
                t_silu_start = time.time()
                act_out = F.silu(gate_out) * up_out
                t_silu_end = time.time()
                t_silu_total += t_silu_end - t_silu_start
                del gate_out, up_out

                # (M, H) = (M, I_b) @ (I_b, H)
                t_down_start = time.time()
                down_out = act_out @ e_down.t()
                t_down_end = time.time()
                t_down_matmul_total += t_down_end - t_down_start
                del act_out

                expert_out += down_out

                # Cleanup references for this bit
                del e_gate_up, e_down, gate_up, down, down_out

            # 最后把结果累加回 final_hidden_states (用 scatter_reduce_)
            t_accum_start = time.time()
            expert_out.mul_(expert_weights)
            final_hidden_states.scatter_reduce_(
                0,
                exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out,
                reduce='sum'
            )
            t_accum_end = time.time()
            t_accum_total += t_accum_end - t_accum_start

            # Cleanup for this expert
            del expert_out, expert_tokens, expert_weights, exp_token_idx

        # Cleanup
        del flat_expert_indices, flat_expert_weights, flat_token_indices
        del idxs, sorted_experts, sorted_weights, sorted_tokens, tokens_per_expert

        t4 = time.time()

        result = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)
        t5 = time.time()

        # 打印详细 profiling (已禁用，仅在需要调试时打开)
        # print(f"  [BitPartitioned_gpu] total={t5-t0:.4f}s | init={t1-t0:.4f}s | shared={t2-t1:.4f}s | router={t3-t2:.4f}s | compute={t4-t3:.4f}s | reshape={t5-t4:.4f}s")
        # print(f"    [Compute_detail] mask={t_mask_total:.4f}s | gate_up_matmul={t_gate_up_matmul_total:.4f}s | silu={t_silu_total:.4f}s | down_matmul={t_down_matmul_total:.4f}s | accum={t_accum_total:.4f}s | active_experts={active_experts_total} | active_bits={self.bit_list}")

        # Optional: Print memory usage
        # mem_str = []
        # if torch.cuda.is_available():
        #     for i in range(torch.cuda.device_count()):
        #         alloc = torch.cuda.memory_allocated(i) / 1024**3
        #         resvd = torch.cuda.memory_reserved(i) / 1024**3
        #         mem_str.append(f"CUDA {i}: {alloc:.2f}GB/{resvd:.2f}GB")
        # print(f"  [MoE Memory] {' | '.join(mem_str)}")

        return result

    def _forward_bit_group(self, x, bit, topk_indices, topk_weights):
        """
        单个 bit group 的前向计算（保留用于测试，用紧凑格式）

        思路：对每个专家，收集所有选了这个专家的 token，然后批量计算
        """
        stats = {
            'mask': 0.0,
            'gate_up_matmul': 0.0,
            'silu': 0.0,
            'down_matmul': 0.0,
            'accum': 0.0,
            'active_experts': 0
        }

        bit_str = str(bit)
        if bit_str not in self.bit_weights.gate_up:
            return torch.zeros_like(x), stats

        out = torch.zeros_like(x)

        gate_up = self.bit_weights.gate_up[bit_str]
        down = self.bit_weights.down[bit_str]
        offsets = self.expert_offsets[bit_str]

        # 对每个 k 位置单独处理
        for k in range(self.top_k):
            expert_indices_k = topk_indices[:, k]  # (N,)
            weights_k = topk_weights[:, k:k+1]  # (N, 1)

            # 对每个专家，收集选了它的 token
            for expert_idx in range(self.num_experts):
                t_mask_start = time.time()
                mask = expert_indices_k == expert_idx
                if not mask.any():
                    continue
                stats['active_experts'] += 1

                token_x = x[mask]  # (M, H)
                token_w = weights_k[mask]  # (M, 1)
                t_mask_end = time.time()
                stats['mask'] += t_mask_end - t_mask_start

                # 获取这个 expert 的实际神经元数
                start = offsets[expert_idx]
                end = offsets[expert_idx + 1]
                actual_inter_size = end - start
                if actual_inter_size == 0:
                    continue

                # 获取这个专家的权重（紧凑格式）
                e_gate_up = gate_up[2*start : 2*end]  # (2*actual_I_b, H)
                e_down = down[:, start:end]           # (H, actual_I_b)

                # 批量计算
                # (M, 2I_b) = (M, H) @ (H, 2I_b)
                t_gate_up_start = time.time()
                gate_up_out = token_x @ e_gate_up.t()
                t_gate_up_end = time.time()
                stats['gate_up_matmul'] += t_gate_up_end - t_gate_up_start

                gate_out = gate_up_out[:, :actual_inter_size]
                up_out = gate_up_out[:, actual_inter_size:]

                # SILU
                t_silu_start = time.time()
                act_out = F.silu(gate_out) * up_out
                t_silu_end = time.time()
                stats['silu'] += t_silu_end - t_silu_start

                # (M, H) = (M, I_b) @ (I_b, H)
                t_down_start = time.time()
                down_out = act_out @ e_down.t()
                t_down_end = time.time()
                stats['down_matmul'] += t_down_end - t_down_start

                # 累加
                t_accum_start = time.time()
                out[mask] += down_out * token_w
                t_accum_end = time.time()
                stats['accum'] += t_accum_end - t_accum_start

        return out, stats
