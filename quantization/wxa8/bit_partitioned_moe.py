#!/usr/bin/env python3
"""WxA8 Bit Partitioned Group MoE —— INT8 激活 + 混合 bit 权重的 MoE。

**存储格式与 WxA16 完全相同**（同一份 packed indices / codebook / norms），
所以这里直接继承 `WxA16BitPartitionedGroupMoE`，只覆盖三处：

  1. `_get_bit_context`  —— 追加每 bit 的 INT8 码本与 cb_step
  2. `_build_hoisted_rotations` —— 预旋转之后顺带量化成 INT8
  3. `_gate_up_matmul` / `_down_matmul` —— 换成 WxA8 kernel

`from_metadata` / `from_build_block` / `_build_group_first` /
`_ensure_active_bits` / `forward` 全部原样复用，checkpoint 加载路径不用改。
低 bit 码本转 int8 近乎无损，**现有 2bpw checkpoint 直接可用，无需重量化**
（见 turboquant_utils/triton_kernels_a8.build_int8_codebook 的说明）。

激活量化的落点（这是 WxA8 唯一真正需要想清楚的地方）：
  - gate_up：旋转矩阵只依赖 (group_size, seed)，与 expert 无关，所以在
    `_build_hoisted_rotations` 里对全量 token 旋转 + 量化一次，per-expert
    只做 int8 切片。顺带把 gather 的字节数减半。
  - down：旋转 seed 含 expert 偏移（seed + original_start），不存在跨 expert
    的复用，只能逐 expert 在旋转后量化。
量化必须在旋转之后 —— 旋转按 group 独立进行，会改变各 group 的幅度分布。
"""

import numpy as np
import torch

from quantization.wxa16.bit_partitioned_moe import WxA16BitPartitionedGroupMoE
from turboquant_utils.rotation import generate_batch_rotation_matrices
from turboquant_utils.triton_kernels_a8 import (
    build_int8_codebook,
    rotate_quantize_fused,
    wxa8_matmul_grouped_slice_in_features_gf,
    wxa8_matmul_grouped_slice_rows_gf,
)


class WxA8BitPartitionedGroupMoE(WxA16BitPartitionedGroupMoE):
    """WxA8 版 MoE。构造/加载与 WxA16 一致，只有 forward 里的两个 matmul 不同。"""

    @classmethod
    def from_wxa16(cls, moe: WxA16BitPartitionedGroupMoE) -> "WxA8BitPartitionedGroupMoE":
        """把已加载的 WxA16 模块原地转成 WxA8。

        因为两者存储格式完全相同，这里只换 __class__，**零张量拷贝、零显存增长**。
        checkpoint 仍然按 WxA16 格式加载（qwen35_quant_io 不用改），
        加载完再调这个方法切到 A8 路径。
        """
        if not isinstance(moe, WxA16BitPartitionedGroupMoE):
            raise TypeError(f"期望 WxA16BitPartitionedGroupMoE，得到 {type(moe)}")
        moe.__class__ = cls
        # bit context 需要重建（要追加 int8 码本）
        moe._bit_ctx_cache = None
        moe._a8_ctx_ready = None
        return moe

    # ---------------------------------------------------------------- context

    def _get_bit_context(self, device):
        """在 WxA16 的 bit context 上追加 INT8 码本与 cb_step。

        cb_step 不折进 norms，而是在量化激活时折进 act_scale ——
        这样权重侧直接复用 WxA16 的 fp16 `norms_gf`，不需要每层再存一份
        fp32 副本（真实规模下 40 层约 1GB）。
        """
        ctx = super()._get_bit_context(device)
        # 基类按 device 缓存并原地返回同一个 dict；用 identity 判断是否已扩展过，
        # 基类因换 device / gf 重建而重造 dict 时会自动重新扩展。
        if getattr(self, "_a8_ctx_ready", None) is ctx:
            return ctx

        for c in ctx.values():
            gu_cb_i8, gu_step = build_int8_codebook(c["gate_up_codebook"])
            dn_cb_i8, dn_step = build_int8_codebook(c["down_codebook"])
            c["gate_up_cb_i8"] = gu_cb_i8
            c["gate_up_cb_step"] = gu_step
            c["down_cb_i8"] = dn_cb_i8
            c["down_cb_step"] = dn_step
            # 激活 dtype：WxA8 kernel 固定产出 fp16，down 的输出要像 WxA16
            # wrapper 那样 cast 回原激活 dtype（真实模型 bf16），否则
            # index_add_ 会 dtype 不匹配。gate.weight 的 dtype 即激活 dtype
            # （warmup_kernels 里同一假设）。
            c["act_dtype"] = self.gate.weight.dtype

        self._a8_ctx_ready = ctx
        return ctx

    # ------------------------------------------------------- hoisted rotation

    def _build_hoisted_rotations(self, x, bit_ctx, tokens_per_expert):
        """预旋转 + 量化成 INT8（覆盖 WxA16 的纯旋转版本）。

        收益判据沿用 WxA16 的逻辑：只对"该 bit 下 expert token 行数之和 >
        T × 阈值"的 bit 做预旋转（长尾 bit 预旋转反而亏）。

        与 WxA16 不同的是，这里的量化让后续 per-expert 的 gather 从 fp16
        变成 int8，字节数减半 —— gather 在 CUDA event 口径下约占 MoE 的 4%。

        Returns:
            {bit_str: (x_i8, x_scale)}，只包含判定为划算的 bit。
            没进这个字典的 bit 会在 `_gate_up_matmul` 里逐 expert 旋转+量化。
        """
        x_src = x if x.dtype == torch.float16 else x.half()

        T = x.shape[0]
        cum = np.asarray(tokens_per_expert)
        counts = np.diff(np.concatenate(([0], cum)))

        out = {}
        for bit_str, ctx in bit_ctx.items():
            if not ctx["gf_gate_up"]:
                continue
            gs = ctx["gate_up_group_size"]
            in_f = ctx["gate_up_in_features"]
            if in_f % gs != 0 or in_f // gs < 2:
                continue

            rows = int(counts[self._expert_mask_by_bit[bit_str]].sum())
            if rows <= T * self.rotation_hoist_threshold:
                continue  # 预旋转不划算（长尾 bit）

            # 旋转 + 量化融合成一个 kernel：x 只读写各一趟，旋转结果不落地
            # （两段式在真实规模下量化那遍单独要 4ms+，融合后整步 0.28ms）
            rot = generate_batch_rotation_matrices(
                gs, ctx["gate_up_seed"], in_f // gs, stride=gs,
                device=x_src.device, dtype=torch.float16)
            out[bit_str] = rotate_quantize_fused(
                x_src, rot, gs, in_f // gs,
                extra_scale=ctx["gate_up_cb_step"])

        return out

    # ------------------------------------------------------------- matmul 覆盖

    def _gate_up_matmul(self, ctx, bit, bit_str, x, x_rot_by_bit,
                        exp_token_idx, expert_tokens, start, end, prof):
        """gate_up 方向的 WxA8 matmul。

        无条件走 A8：实测小 B 下 A8 最多慢 2.8us（B=8 时），而大 B 能省
        26.6us；为小 B 维护 fp16 缓冲（每 bit 约 128MB）或逐 expert 重算
        旋转（额外 launch ~5us）都比直接吃掉那 ≤2.8us 更亏
        （见 test/test_wxa8_kernel.py --crossover 与 triton_kernels_a8 的
        WXA8_MIN_B 注释）。
        """
        gu = ctx["gate_up"]
        gs = ctx["gate_up_group_size"]

        hoisted = x_rot_by_bit.get(bit_str)
        if hoisted is not None:
            x_i8_all, x_scale_all = hoisted
            with prof.stage("gather_rotated"):
                inp_i8 = x_i8_all[exp_token_idx]
                inp_scale = x_scale_all[exp_token_idx]
        else:
            # 该 bit 没做预旋转（长尾 bit）：逐 expert 旋转 + 量化（融合 kernel）
            if expert_tokens is None:
                with prof.stage("gather_x"):
                    expert_tokens = x[exp_token_idx]
            with prof.stage("act_quant"):
                src = expert_tokens
                if src.dtype != torch.float16:
                    src = src.half()
                in_f = ctx["gate_up_in_features"]
                rot = generate_batch_rotation_matrices(
                    gs, ctx["gate_up_seed"], in_f // gs, stride=gs,
                    device=src.device, dtype=torch.float16)
                inp_i8, inp_scale = rotate_quantize_fused(
                    src, rot, gs, in_f // gs,
                    extra_scale=ctx["gate_up_cb_step"])

        with prof.stage("gate_up_kernel"):
            gate_up_out = wxa8_matmul_grouped_slice_rows_gf(
                inp_i8, inp_scale,
                gu["indices_packed_gf"],
                ctx["gate_up_cb_i8"],
                gu["norms_gf"],
                gs,
                ctx["gate_up_in_features"],
                2 * start, 2 * end,  # row slice
                bit,
                norms_prescaled=ctx["gate_up_norms_prescaled"],
            )
        del inp_i8, inp_scale
        return gate_up_out, expert_tokens

    def _down_matmul(self, ctx, bit, act_out, start, end, prof):
        """down 方向的 WxA8 matmul。

        down 的旋转 seed 含 expert 偏移（seed + original_start），无法跨
        expert 复用，所以旋转 + 量化只能在这里逐 expert 做。
        """
        dn = ctx["down"]
        gs = ctx["down_group_size"]
        num_groups = (end - start) // gs

        with prof.stage("act_quant_down"):
            src = act_out if act_out.dtype == torch.float16 else act_out.half()
            rot = generate_batch_rotation_matrices(
                gs, ctx["down_seed"] + start, num_groups, stride=gs,
                device=src.device, dtype=torch.float16)
            act_i8, act_scale = rotate_quantize_fused(
                src, rot, gs, num_groups,
                extra_scale=ctx["down_cb_step"])

        with prof.stage("down_kernel"):
            down_out = wxa8_matmul_grouped_slice_in_features_gf(
                act_i8, act_scale,
                dn["indices_packed_gf"],
                ctx["down_cb_i8"],
                dn["norms_gf"],
                gs,
                start, end,  # original_start, original_end
                bit,
                norms_prescaled=ctx["down_norms_prescaled"],
            )
        del act_i8, act_scale
        # 对齐 WxA16 wrapper 的语义：输出 cast 回原激活 dtype（真实模型 bf16）
        if down_out.dtype != ctx["act_dtype"]:
            down_out = down_out.to(ctx["act_dtype"])
        return down_out
