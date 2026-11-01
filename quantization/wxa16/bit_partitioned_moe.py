#!/usr/bin/env python3
"""
WxA16 Bit Partitioned Group MoE - 存储 packed 量化权重的 MoE 模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import math
import os
import time
import gc

import numpy as np

from .linear import WxA16Linear
from quantization.common.adapters import router_forward, shared_forward
from turboquant_utils.quantize import turboquant_quantize_packed_full
from turboquant_utils.triton_kernels import (
    triton_fused_matmul_grouped_slice_rows,
    triton_fused_matmul_grouped_slice_in_features,
    triton_fused_matmul_grouped_slice_rows_gf,
    triton_fused_matmul_grouped_slice_in_features_gf,
    _get_fused_grouped_config,
    convert_to_group_first,
)
from turboquant_utils.cuda_profiler import CudaStageProfiler
from turboquant_utils.rotation import (
    batch_rotate_input,
    generate_batch_rotation_matrices,
)


class WxA16Weights(nn.Module):
    """
    单个 bit 宽度的权重存储。

    存储 gate_up 和 down 的 packed 量化表示。
    """

    def __init__(
        self,
        bit_width: int,
        hidden_size: int,
    ):
        super().__init__()
        self.bit_width = bit_width
        self.hidden_size = hidden_size

        # 实际的量化参数会在 later 填充
        self._gate_up_packed = None
        self._down_packed = None

        # 保存元数据（非 tensor 值）
        self._gate_up_seed = None
        self._gate_up_group_size = None
        self._gate_up_shape = None
        self._gate_up_bit_width = None
        self._gate_up_rotation = None
        self._gate_up_orig_dtype = None

        self._down_seed = None
        self._down_group_size = None
        self._down_shape = None
        self._down_bit_width = None
        self._down_rotation = None
        self._down_orig_dtype = None

    @property
    def gate_up_packed(self):
        """获取 gate_up_packed 字典，同时恢复元数据"""
        if self._gate_up_packed is None:
            # 从所有 gate_up_* 开头的 buffer 重建字典
            self._gate_up_packed = {}
            for name, param in self.named_buffers():
                if name.startswith('gate_up_'):
                    key = name[len('gate_up_'):]
                    self._gate_up_packed[key] = param
            # 恢复元数据
            if self._gate_up_seed is not None:
                self._gate_up_packed['seed'] = self._gate_up_seed
            if self._gate_up_group_size is not None:
                self._gate_up_packed['group_size'] = self._gate_up_group_size
            if self._gate_up_shape is not None:
                self._gate_up_packed['shape'] = self._gate_up_shape
            if self._gate_up_bit_width is not None:
                self._gate_up_packed['bit_width'] = self._gate_up_bit_width
            if self._gate_up_rotation is not None:
                self._gate_up_packed['rotation'] = self._gate_up_rotation
            if self._gate_up_orig_dtype is not None:
                self._gate_up_packed['orig_dtype'] = self._gate_up_orig_dtype
        return self._gate_up_packed

    @gate_up_packed.setter
    def gate_up_packed(self, value):
        self._gate_up_packed = value

    @property
    def down_packed(self):
        """获取 down_packed 字典，同时恢复元数据"""
        if self._down_packed is None:
            # 从所有 down_* 开头的 buffer 重建字典
            self._down_packed = {}
            for name, param in self.named_buffers():
                if name.startswith('down_'):
                    key = name[len('down_'):]
                    self._down_packed[key] = param
            # 恢复元数据
            if self._down_seed is not None:
                self._down_packed['seed'] = self._down_seed
            if self._down_group_size is not None:
                self._down_packed['group_size'] = self._down_group_size
            if self._down_shape is not None:
                self._down_packed['shape'] = self._down_shape
            if self._down_bit_width is not None:
                self._down_packed['bit_width'] = self._down_bit_width
            if self._down_rotation is not None:
                self._down_packed['rotation'] = self._down_rotation
            if self._down_orig_dtype is not None:
                self._down_packed['orig_dtype'] = self._down_orig_dtype
        return self._down_packed

    @down_packed.setter
    def down_packed(self, value):
        self._down_packed = value

    def set_packed_data(self, gate_up_packed, down_packed):
        """
        设置 packed 数据并注册为 buffer，以便 state_dict 保存。

        注意：group-first 布局由 property getter 惰性生成（首次访问时），
        不注册为 buffer，不影响 state_dict 格式。
        """
        # Register gate_up packed data
        if gate_up_packed is not None:
            for key, value in gate_up_packed.items():
                if isinstance(value, torch.Tensor):
                    self.register_buffer(f"gate_up_{key}", value)
            self._gate_up_packed = gate_up_packed
            # 保存元数据
            self._gate_up_seed = gate_up_packed.get('seed')
            self._gate_up_group_size = gate_up_packed.get('group_size')
            self._gate_up_shape = gate_up_packed.get('shape')
            self._gate_up_bit_width = gate_up_packed.get('bit_width')
            self._gate_up_rotation = gate_up_packed.get('rotation')
            self._gate_up_orig_dtype = gate_up_packed.get('orig_dtype')

        # Register down packed data
        if down_packed is not None:
            for key, value in down_packed.items():
                if isinstance(value, torch.Tensor):
                    self.register_buffer(f"down_{key}", value)
            self._down_packed = down_packed
            # 保存元数据
            self._down_seed = down_packed.get('seed')
            self._down_group_size = down_packed.get('group_size')
            self._down_shape = down_packed.get('shape')
            self._down_bit_width = down_packed.get('bit_width')
            self._down_rotation = down_packed.get('rotation')
            self._down_orig_dtype = down_packed.get('orig_dtype')

    def _build_group_first(self, which: str = "both", offload_original_to_cpu: bool = True):
        """P5-1: 生成 group-first 布局的权重（惰性，首次 forward 前调用）。

        不注册为 buffer，仅作为运行时优化产物存在字典中，
        不影响 state_dict 格式和 checkpoint 兼容性。

        若 _packed 字典为 None（from_metadata 路径下），先通过 property 触发重建。

        Args:
            which: "gate_up" / "down" / "both"
            offload_original_to_cpu: 构建完 gf 后将原始 indices_packed/norms 移到 CPU，
                释放 GPU 显存（gf 与原始数据元素总数相同，显存净占用不变）。
        """
        if which in ("gate_up", "both"):
            packed = self.gate_up_packed  # 触发惰性重建
            if "indices_packed_gf" not in packed:
                gu_indices_gf, gu_norms_gf = convert_to_group_first(
                    packed["indices_packed"],
                    packed["norms"],
                    packed["group_size"],
                    packed["bit_width"],
                )
                packed["indices_packed_gf"] = gu_indices_gf
                # P6-2: 把常量缩放 1/sqrt(group_size) 预乘进 norms_gf，
                # 省掉 kernel wrapper 里每次调用一个微 kernel（每层约 512 次）。
                # 原始 packed["norms"] 保持未缩放，state_dict 格式不受影响。
                packed["norms_gf"] = gu_norms_gf / math.sqrt(packed["group_size"])
                packed["norms_gf_prescaled"] = True

                if offload_original_to_cpu:
                    # 原始 buffer 移到 CPU，释放 GPU 显存
                    # state_dict() 仍能正确保存（buffer 还是 module 的一部分）
                    # forward 只走 gf 路径，不再访问原始布局
                    self.gate_up_indices_packed = self.gate_up_indices_packed.cpu()
                    self.gate_up_norms = self.gate_up_norms.cpu()
                    # 同步更新字典引用
                    packed["indices_packed"] = self.gate_up_indices_packed
                    packed["norms"] = self.gate_up_norms
        if which in ("down", "both"):
            packed = self.down_packed  # 触发惰性重建
            if "indices_packed_gf" not in packed:
                dn_indices_gf, dn_norms_gf = convert_to_group_first(
                    packed["indices_packed"],
                    packed["norms"],
                    packed["group_size"],
                    packed["bit_width"],
                )
                packed["indices_packed_gf"] = dn_indices_gf
                # P6-2: 同 gate_up，预乘常量缩放
                packed["norms_gf"] = dn_norms_gf / math.sqrt(packed["group_size"])
                packed["norms_gf_prescaled"] = True

                if offload_original_to_cpu:
                    self.down_indices_packed = self.down_indices_packed.cpu()
                    self.down_norms = self.down_norms.cpu()
                    packed["indices_packed"] = self.down_indices_packed
                    packed["norms"] = self.down_norms


    @classmethod
    def from_metadata(cls, bit_width: int, hidden_size: int,
                      gate_up_meta: dict, down_meta: dict) -> "WxA16Weights":
        """
        从保存的元数据重建 WxA16Weights（checkpoint 加载路径）。

        与量化路径共用 set_packed_data：注册占位 buffer + 恢复元数据属性，
        实际数据由 load_state_dict(assign=True) 回填。
        """
        def _placeholder_pack(meta: dict) -> dict:
            return {
                "indices_packed": torch.empty(meta["indices_packed_shape"], dtype=torch.uint8),
                "codebook": torch.empty(meta["codebook_shape"], dtype=torch.float16),
                "norms": torch.empty(meta["norms_shape"], dtype=torch.float16),
                "seed": meta["seed"],
                "group_size": meta["group_size"],
                "shape": tuple(meta["shape"]),
                "bit_width": meta["bit_width"],
                "rotation": meta["rotation"],
                "orig_dtype": meta["orig_dtype"],
            }

        weights = cls(bit_width, hidden_size)
        weights.set_packed_data(_placeholder_pack(gate_up_meta), _placeholder_pack(down_meta))
        # 清空缓存字典：load_state_dict(assign=True) 只替换注册的 buffer，
        # 字典里仍是占位张量，置 None 让 gate_up_packed/down_packed 属性
        # 在回填后从 named_buffers 惰性重建，保证 forward 读到真实数据
        weights._gate_up_packed = None
        weights._down_packed = None
        return weights


# ---------------------------------------------------------------------------
# 静态派发（CUDA graph decode，自 DartMoQ-Qwen3.5 移植，261029）
#
# MoE 前向的 host 编排（token 统计读回 host → Python 建池字典 → 变长拼接
# → host 构建调度表传设备）与图模式根本冲突：capture 会把捕获那一刻的
# 路由烤死进图。静态化三条支柱（kernel 零改动）：
#   1. merged kernel 设备驱动——专家行段与块调度表运行时从设备张量读取，
#      调度表按最坏情况静态化（空 expert 的块全掩码安全退出）；
#   2. decode 每层总行数恒等于 top-k → 固定容量池缓冲 + 设备端统计
#      （index_add）/池内游标（cumsum）/行内位次（定形比较算子）组装，
#      padding 行置零、权重为零的 scatter 对合法 token 恒等加零；
#   3. gate_up 旋转 decode 下无条件预提升（token 数 1，数学等价）；down
#      逐 expert 旋转改为按 bit 全局预建旋转矩阵库 + 设备索引选取。
# 本次移植仅接通 int8（WxA8 的 _static_pool_compute 已随 triton_kernels_a8
# 的 merged/static kernel 一起移植）；基类声明 _STATIC_DISPATCH_READY=False，
# 未实现静态池的模式被 forward 钩子挡掉（防把基类数值烤进图）。
# ---------------------------------------------------------------------------

_STATIC_DISPATCH_ENV = "DARTMOQ_STATIC_DISPATCH"
_STATIC_DISPATCH_MAX_CAPACITY = 64


class _StaticPoolState:
    """单 bit 池的静态派发常量与容量缓冲（CUDA graph decode 用）。

    成员分两类：加载期常量（expert_info 的列切片列、最坏情况调度表、
    kernel 入参、旋转矩阵库引用）与容量上界固定的复用缓冲（tokens/
    weights/slot_local 缓冲、expert_info 的行段列）。每步前向由设备端
    定形算子原地重写后者，无 host 读取、无 H2D、无形状变化——capture
    期合法；重放期路由变化通过行段列（token_start/token_count）体现，
    kernel 运行时读取。
    """

    def __init__(self, bit_str, members, ctx, capacity, act_dtype,
                 cfg_getter=None, shared_rot_bank=None):
        # members: [(expert_idx, start, end)]，按 expert 升序（与 _merged_pools
        # 的成员顺序一致）；start/end 是该 expert 在本 bit packed 布局上的段。
        # DartMoQ 移植：池按 (bit, inter) 划分，members 的 inter 恒一致
        #（_build_static_decode_state 已拆好；跨 bit 的段在本池外，max(starts)
        # 只算本池成员）。shared_rot_bank：同 bit 各子池共享的 down 旋转库
        #（None 则按本池 extent 自建——独立构造路径保留）
        # cfg_getter: direction/B → tile 配置（各模式 kernel 的选档函数不同，
        # 由 MoE 实例按 bit 提供；None 回退 A16 的 _get_fused_grouped_config）
        self.bit_str = bit_str
        self.bit = int(bit_str)
        self.capacity = capacity
        self.num_members = len(members)
        starts = [s for _e, s, _t in members]
        inter = members[0][2] - members[0][1]

        gs_gu = ctx["gate_up_group_size"]
        gs_dn = ctx["down_group_size"]
        in_f = ctx["gate_up_in_features"]
        if not ctx["gf_gate_up"] or not ctx["gf_down"]:
            raise RuntimeError(
                f"static dispatch: bit {bit_str} lacks group-first layout")
        if in_f % gs_gu != 0 or in_f // gs_gu < 2:
            raise RuntimeError(
                f"static dispatch: bit {bit_str} gate_up num_groups < 2")
        if inter % gs_dn != 0:
            raise RuntimeError(
                f"static dispatch: bit {bit_str} down inter not group-aligned")
        # down 允许 g_count=1（inter == gs）：DartMoQ 的混合 bit 分配有
        # inter=128 的 expert（gs=128），merged kernel 的 group 循环
        # （for g in range(NUM_GROUPS)）与 pooled 旋转 kernel 的 grid 维
        # 对 1 都合法；eager 路径的 slice_in_features kernel 同样跑过
        # num_groups=1（test_dsv2_native_cache 的 decode 全通）
        if any(s % gs_dn for s in starts):
            raise RuntimeError(
                f"static dispatch: bit {bit_str} down start not group-aligned")
        self.gs_gu, self.gs_dn = gs_gu, gs_dn
        self.in_f = in_f
        self.inter = inter
        self.g_count = inter // gs_dn
        self.n_slice = 2 * inter
        self.n_limit = 2 * max(starts) + self.n_slice

        device = ctx["gate_up_codebook"].device
        self.member_ids = torch.tensor([e for e, _s, _t in members],
                                       dtype=torch.long, device=device)

        # expert_info 模板：(token_start, token_count, slice_start, slice_count)
        # 行段列（0,1）每步设备端重写；列切片列（2,3）常量
        ei_gu = torch.zeros(self.num_members, 4, dtype=torch.int32, device=device)
        ei_gu[:, 2] = torch.tensor([2 * s for s in starts],
                                   dtype=torch.int32, device=device)
        ei_gu[:, 3] = self.n_slice
        ei_dn = torch.zeros(self.num_members, 4, dtype=torch.int32, device=device)
        ei_dn[:, 2] = torch.tensor([s // gs_dn for s in starts],
                                   dtype=torch.int32, device=device)
        ei_dn[:, 3] = self.g_count
        self.expert_info_gu, self.expert_info_dn = ei_gu, ei_dn
        self.start_gs_lut = ei_dn[:, 2].to(torch.long)

        # kernel 入参常量（与 bit_ctx 同源，避免每 forward 字典查找）
        gu, dn = ctx["gate_up"], ctx["down"]
        self.idx_gu_gf = gu["indices_packed_gf"]
        self.norms_gu_gf = gu["norms_gf"]
        self.cb_gu = ctx["gate_up_codebook"]
        self.prescaled_gu = ctx["gate_up_norms_prescaled"]
        self.idx_dn_gf = dn["indices_packed_gf"]
        self.norms_dn_gf = dn["norms_gf"]
        self.cb_dn = ctx["down_codebook"]
        self.prescaled_dn = ctx["down_norms_prescaled"]
        self.gu_seed = ctx["gate_up_seed"]
        self.N_dn = self.norms_dn_gf.shape[1]

        # 各模式的 kernel 入参常量：ctx 里存在则抓引用。A16 路径用上面的
        # cb_gu/cb_dn；wxa8 的 cb_step 常量与 int8 码本由子类的
        # _static_pool_compute 读取。
        # gate_up 旋转每池统一（seed 无 expert 偏移），此处一次生成整批矩阵
        self.cb_step_gu = ctx.get("gate_up_cb_step", 1.0)
        self.cb_scale_gu = ctx.get("gate_up_cb_scale", 1.0)
        self.cb_step_dn = ctx.get("down_cb_step", 1.0)
        self.cb_scale_dn = ctx.get("down_cb_scale", 1.0)
        self.cb_gu_i8 = ctx.get("gate_up_cb_i8")
        self.cb_dn_i8 = ctx.get("down_cb_i8")
        self.rot_gu = generate_batch_rotation_matrices(
            gs_gu, ctx["gate_up_seed"], in_f // gs_gu, stride=gs_gu,
            device=device, dtype=torch.float16)

        # launch 配置与最坏情况调度表：每 expert 全 B-block × 全 N-block，
        # 空专家的块由 kernel 的行掩码空转（安全性见 kernel 的 mask_b 逻辑）
        if cfg_getter is None:
            def cfg_getter(direction, B, _bit=self.bit):
                return _get_fused_grouped_config(
                    _bit, direction=direction, B=B)
        self.cfg_gu = cfg_getter("gate_up", capacity)
        self.cfg_dn = cfg_getter("down", capacity)
        self.block_info_gu = self._worst_case_blocks(
            capacity, self.cfg_gu[0], self.cfg_gu[1], self.n_slice)
        self.block_info_dn = self._worst_case_blocks(
            capacity, self.cfg_dn[0], self.cfg_dn[1], self.N_dn)
        # 行块调度表（pooled 旋转量化 kernel 用）：BLOCK_B 恒 32
        self.block_info_b32 = self._b32_blocks(capacity)

        # down 旋转矩阵库：seed = down_seed + j*gs，j ∈ [0, 池总 group 数)。
        # 逐 expert 路径的 batch_rotate_input(seed + start) 与此同一索引空间
        # （start 是 gs 的倍数），故 bank 按全局 seed 序展开即可——复用批量
        # 旋转缓存，跨层同 (gs, seed) 的池共享同一份 bank。
        # DartMoQ 子池：同 bit 各子池共用一份覆盖整个 bit extent 的库
        #（kernel 的 rot_base 是全局 group 基序号，直接索引；逐子池按各自
        # extent 建会在缓存里按 num_groups 分键存多份重叠副本）
        need_groups = (max(starts) + inter) // gs_dn
        if shared_rot_bank is not None:
            if shared_rot_bank.shape[0] < need_groups:
                raise RuntimeError(
                    f"static dispatch: shared rot_bank too small "
                    f"({shared_rot_bank.shape[0]} < {need_groups})")
            self.rot_bank = shared_rot_bank
        else:
            self.rot_bank = generate_batch_rotation_matrices(
                gs_dn, ctx["down_seed"], need_groups, stride=gs_dn,
                device=device, dtype=torch.float16)
        self.j_arange = torch.arange(self.g_count, dtype=torch.long, device=device)

        # 容量缓冲（C+1：最后一格是丢弃槽——非本池行写入它，永不读出，
        # 保证 index_copy_ 的目标下标唯一、结果确定）
        C = capacity
        self.tokens_buf = torch.zeros(C + 1, dtype=torch.long, device=device)
        self.weights_buf = torch.zeros(C + 1, dtype=act_dtype, device=device)
        self.slot_local_buf = torch.zeros(C + 1, dtype=torch.long, device=device)

    def _worst_case_blocks(self, capacity, block_b, block_n, n_extent):
        """最坏情况调度表：(E × ceil(C/BLOCK_B) × ceil(N/BLOCK_N), 3) int32。"""
        n_b = (capacity + block_b - 1) // block_b
        n_n = (n_extent + block_n - 1) // block_n
        blocks = [(e, b, n) for e in range(self.num_members)
                  for b in range(n_b) for n in range(n_n)]
        return torch.tensor(blocks, dtype=torch.int32,
                            device=self.member_ids.device).contiguous()

    def _b32_blocks(self, capacity):
        """行块静态调度表（pooled 旋转量化 kernel 用）：(E × ceil(C/32), 2) int32。"""
        n_b = (capacity + 31) // 32
        blocks = [(e, b) for e in range(self.num_members) for b in range(n_b)]
        return torch.tensor(blocks, dtype=torch.int32,
                            device=self.member_ids.device).contiguous()


class _StaticDispatchState:
    """一层 MoE 的静态派发总状态：容量上界 + 池列表 + expert → 池/成员 LUT。"""

    def __init__(self, capacity, pools, expert_bit, expert_local):
        self.capacity = capacity
        self.pools = pools
        self.expert_bit = expert_bit      # (num_experts,) long：expert → 池序号
        self.expert_local = expert_local  # (num_experts,) long：expert → 池内成员序号
        dev = expert_bit.device
        self.ones_c = torch.ones(capacity, dtype=torch.int32, device=dev)
        self.arange_c = torch.arange(capacity, dtype=torch.int64, device=dev)


class WxA16BitPartitionedGroupMoE(nn.Module):
    """
    按 bit 分区的 WxA16 量化 MoE。

    存储格式:
      - 每个 bit 有自己的 packed gate_up/down 权重
      - 紧凑存储，无冗余
    """

    def __init__(
        self,
        gate,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int = 6,
        shared_expert = None,
        shared_expert_gate = None,
        shared_experts = None,
        router_mode: str = "softmax_topk",
        shared_mode: str = "none",
        mlp_returns_tuple: bool = False,
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
        # behavioral conventions (quantization/common/adapters.py), written to
        # meta.json and restored on load
        self.router_mode = router_mode
        self.shared_mode = shared_mode
        # olmoe's DecoderLayer expects the mlp to return a
        # (hidden_states, router_logits) 2-tuple
        self.mlp_returns_tuple = mlp_returns_tuple

        # 计时开关：开启时 forward 会测量各阶段时间并存入 self.last_timings
        # 关闭时跳过所有计时采样和 print，零额外开销
        self.enable_timing = True
        self.last_timings = {}

        # P6-0: CUDA event 分阶段计时（与上面的 wall-clock 计时并存，互不影响）
        # wall-clock 计时测的是 CPU launch 时间，异步执行下会把耗时记错段；
        # 这里用 event 测真实 GPU 执行时间。默认关闭，测量时由外部置 True。
        self.enable_cuda_profile = False
        self._cuda_prof = CudaStageProfiler(enabled=False)
        self.last_cuda_stats = {}

        # P6-1: gate_up 旋转预提升开关。
        # gate_up 的旋转矩阵只依赖 (group_size, seed)，与 expert 无关，
        # 而 top_k=8 导致每个 token 在 per-expert 循环里被重复旋转 top_k 次。
        # 开启后按 bit 判定收益（见 forward 内的判据），划算才预旋转。
        self.enable_rotation_hoist = True
        # 预旋转收益门限：该 bit 下所有 expert 的 token 行数之和 > T * 该系数 才预旋转。
        # 取 1.0 即"预旋转的额外成本(旋转 T 行) 小于省下的重复旋转"这一盈亏平衡点，
        # 留一点余量避免边界抖动。
        self.rotation_hoist_threshold = 1.5

        # 按 bit 分组的权重
        self.bit_weights = nn.ModuleDict()  # "8" -> WxA16Weights, "4" -> ...

        # expert 位置信息
        self.inter_size_by_bit = {}
        self.expert_offsets = {}  # bit_str -> LongTensor (GPU)
        self._expert_offsets_cpu = {}  # bit_str -> List[int] (CPU 常驻，避免每 forward .item() D2H 同步)

        self.bit_list = []

    @classmethod
    @torch.no_grad()
    def from_build_block(cls, build_block, layer_metadata, group_size: int = 128):
        """
        从 MoEBuildBlock（DartMoQ 侧为重建后的 hybrid MoE）重构为 WxA16BitPartitionedGroupMoE。

        Args:
            build_block: 重建后的 hybrid MoE (包含 DartMoQHybridWrapper，权重为原始 fp16)
            layer_metadata: 量化过程中的元数据（含 router_mode/shared_mode/top_k）
            group_size: TurboQuant 分组大小

        Returns:
            WxA16BitPartitionedGroupMoE 实例
        """
        tick_start = time.time()
        from bit_partitioned_moe import BitPartitionedGroupMoE

        # 先从 build_block 构建普通的 BitPartitionedGroupMoE（提取原始 fp16 权重）
        tick_fp16moe = time.time()
        fp16_moe = BitPartitionedGroupMoE.from_build_block(build_block, layer_metadata)
        print(f"  [DEBUG] BitPartitionedGroupMoE.from_build_block: {time.time() - tick_fp16moe:.4f}s")

        # behavioral conventions and top_k come from layer_metadata first
        # (single source for both the build and load sides)
        router_mode = layer_metadata.get("router_mode", "softmax_topk")
        shared_mode = layer_metadata.get("shared_mode", "none")
        top_k = layer_metadata.get("top_k") or getattr(fp16_moe, "top_k", 6)
        mlp_returns_tuple = layer_metadata.get("mlp_returns_tuple", False)

        # 创建 WxA16 版本
        tick_create = time.time()
        moe = cls(
            gate=fp16_moe.gate,
            num_experts=fp16_moe.num_experts,
            hidden_size=fp16_moe.hidden_size,
            intermediate_size=fp16_moe.intermediate_size,
            top_k=top_k,
            shared_expert=getattr(fp16_moe, "shared_expert", None),
            shared_expert_gate=getattr(fp16_moe, "shared_expert_gate", None),
            shared_experts=getattr(fp16_moe, "shared_experts", None),
            router_mode=router_mode,
            shared_mode=shared_mode,
            mlp_returns_tuple=mlp_returns_tuple,
        )
        print(f"  [DEBUG] Create WxA16BitPartitionedGroupMoE instance: {time.time() - tick_create:.4f}s")

        moe.bit_list = fp16_moe.bit_list

        # 获取 dtype 和 device
        dtype = next(fp16_moe.parameters()).dtype if hasattr(fp16_moe, 'parameters') else torch.float16
        device = next(build_block.parameters()).device

        # 对每个 bit 的权重进行 WxA16 量化
        tick_quantize = time.time()
        for bit_str, gate_up_weight in fp16_moe.bit_weights.gate_up.items():
            bit = int(bit_str)
            tick_bit = time.time()

            # defensive skip: 0-bit means pruned (no sub-experts exist, the
            # bucket would be all zeros), and packed quantization does not
            # support 0 bit. The normal flow already excludes 0 from bit_list.
            if bit == 0:
                print(f"  Skip quantizing bit 0 (pruned, no sub-experts)")
                continue

            print(f"  Quantizing MoE weights for {bit} bit...")

            # 量化 gate_up
            down_weight = fp16_moe.bit_weights.down[bit_str]

            # 由于 gate_up 是拼接的权重 (2x_neurons, hidden_size)，我们需要特殊处理
            # gate_proj 和 up_proj 是拼接的
            tick_gateup = time.time()
            gate_up_packed = turboquant_quantize_packed_full(
                gate_up_weight.data,
                bit_width=bit,
                group_size=group_size,
                seed=42 + bit,
                keep_on_gpu=True,
            )
            print(f"    [DEBUG] turboquant_quantize_packed_full (gate_up): {time.time() - tick_gateup:.4f}s")

            tick_down = time.time()
            down_packed = turboquant_quantize_packed_full(
                down_weight.data,
                bit_width=bit,
                group_size=group_size,
                seed=42 + bit + 1000,
                keep_on_gpu=True,
            )
            print(f"    [DEBUG] turboquant_quantize_packed_full (down): {time.time() - tick_down:.4f}s")

            # 创建 WxA16Weights
            tick_weights = time.time()
            wxa16_weights = WxA16Weights(bit, moe.hidden_size)
            print(f"    [DEBUG] Create WxA16Weights: {time.time() - tick_weights:.4f}s")

            # 存储 packed 数据并注册为 buffer
            tick_setpacked = time.time()
            wxa16_weights.set_packed_data(gate_up_packed, down_packed)
            print(f"    [DEBUG] set_packed_data: {time.time() - tick_setpacked:.4f}s")

            # 复制 offset 信息
            moe.expert_offsets[bit_str] = fp16_moe.expert_offsets[bit_str]
            # CPU 常驻副本：每 expert×每 bit 两次 .item() 会触发 D2H 同步，
            # 提前转成 Python list 后循环里直接读，零同步开销
            moe._expert_offsets_cpu[bit_str] = fp16_moe.expert_offsets[bit_str].cpu().tolist()
            moe.inter_size_by_bit[bit] = fp16_moe.inter_size_by_bit[bit]

            moe.bit_weights[bit_str] = wxa16_weights

            print(f"  Done quantizing {bit} bit: {time.time() - tick_bit:.4f}s")

        print(f"  [DEBUG] Quantize all bits: {time.time() - tick_quantize:.4f}s")

        # 清理 fp16_moe
        tick_cleanup = time.time()
        del fp16_moe
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  [DEBUG] Cleanup fp16_moe: {time.time() - tick_cleanup:.4f}s")

        print(f"  [DEBUG] from_build_block total time: {time.time() - tick_start:.4f}s")
        return moe

    @classmethod
    def from_metadata(cls, meta: dict, gate, shared_expert=None,
                      shared_expert_gate=None, shared_experts=None) -> "WxA16BitPartitionedGroupMoE":
        """
        从保存的元数据重建 WxA16BitPartitionedGroupMoE（checkpoint 加载路径）。

        复用传入的 gate/shared_expert/shared_expert_gate/shared_experts（与量化路径
        复用同一批对象，保证 state_dict key 结构一致），bit_weights 按保存顺序重建，
        expert_offsets / _expert_offsets_cpu 从 JSON list 恢复。
        router_mode/shared_mode 从 meta 恢复（旧 checkpoint 无此字段时回落通用约定）。
        """
        moe = cls(
            gate=gate,
            num_experts=meta["num_experts"],
            hidden_size=meta["hidden_size"],
            intermediate_size=meta["intermediate_size"],
            top_k=meta["top_k"],
            shared_expert=shared_expert,
            shared_expert_gate=shared_expert_gate,
            shared_experts=shared_experts,
            router_mode=meta.get("router_mode", "softmax_topk"),
            shared_mode=meta.get("shared_mode", "gated" if shared_expert is not None else "none"),
            mlp_returns_tuple=meta.get("mlp_returns_tuple", False),
        )

        moe.bit_list = list(meta["bit_list"])
        moe.inter_size_by_bit = {int(k): v for k, v in meta["inter_size_by_bit"].items()}
        moe.expert_offsets = {
            bit_str: torch.tensor(lst, dtype=torch.long) for bit_str, lst in meta["expert_offsets"].items()
        }
        moe._expert_offsets_cpu = {
            bit_str: list(lst) for bit_str, lst in meta["expert_offsets"].items()
        }

        # 按 JSON 保序插入（= 保存时 ModuleDict 顺序 = forward 的 bit 处理顺序）
        for bit_str, bit_meta in meta["bits"].items():
            moe.bit_weights[bit_str] = WxA16Weights.from_metadata(
                int(bit_str), meta["hidden_size"], bit_meta["gate_up"], bit_meta["down"]
            )

        moe.enable_timing = meta.get("enable_timing", True)
        return moe

    def _ensure_active_bits(self):
        """P6-2: 预计算每个 expert 真正活跃的 bit 列表。

        原先 forward 内层是 `for bit_str in self.bit_weights.keys()` 全遍历，
        再靠 `if actual_inter_size == 0: continue` 跳过。实测本模型 bit 是在
        expert 之间划分的（每个 expert 基本只用一个 bit），256 experts × 3 bits
        里约 512 次是纯空转。

        只依赖 expert_offsets（加载后固定），算一次复用。
        """
        if getattr(self, "_active_bits_cache_valid", False):
            return

        table = []
        for e in range(self.num_experts):
            entries = []
            for bit_str in self.bit_weights.keys():
                offs = self._expert_offsets_cpu[bit_str]
                s, t = offs[e], offs[e + 1]
                if t > s:
                    entries.append((bit_str, int(bit_str), s, t))
            table.append(entries)
        self._active_bits_by_expert = table

        # 每个 bit 下哪些 expert 活跃（P6-1 的收益判据要用）
        masks = {}
        for bit_str in self.bit_weights.keys():
            arr = np.asarray(self._expert_offsets_cpu[bit_str])
            masks[bit_str] = (arr[1:] - arr[:-1]) > 0
        self._expert_mask_by_bit = masks

        self._active_bits_cache_valid = True

    def _get_bit_context(self, device):
        """P6-2: 缓存每个 bit 的常量，避免在 per-expert × per-bit 循环里重复取。

        原先每层约 512 次重复做：dict 查找、`codebook.to(device)`、
        以及 kernel wrapper 内部的 `seed.item()`（若 seed 是 tensor，这是 D2H 同步）。
        这些量在一层内是常量，缓存一次即可。
        """
        cache = getattr(self, "_bit_ctx_cache", None)
        if cache is not None and getattr(self, "_bit_ctx_device", None) == device:
            return cache

        def _as_int(v):
            return int(v.item()) if torch.is_tensor(v) else int(v)

        ctx = {}
        for bit_str, w in self.bit_weights.items():
            gu = w.gate_up_packed
            dn = w.down_packed
            ctx[bit_str] = {
                "gate_up": gu,
                "down": dn,
                # gate_up / down 的 gf 布局是各自独立构建的，分别判断（保持原语义）
                "gf_gate_up": "indices_packed_gf" in gu,
                "gf_down": "indices_packed_gf" in dn,
                "gate_up_codebook": gu["codebook"].to(device),
                "down_codebook": dn["codebook"].to(device),
                "gate_up_seed": _as_int(gu["seed"]),
                "down_seed": _as_int(dn["seed"]),
                "gate_up_group_size": int(gu["group_size"]),
                "down_group_size": int(dn["group_size"]),
                "gate_up_in_features": int(gu["shape"][1]),   # = hidden_size
                "down_in_features": int(dn["shape"][1]),      # = full_in_features
                "gate_up_norms_prescaled": bool(gu.get("norms_gf_prescaled", False)),
                "down_norms_prescaled": bool(dn.get("norms_gf_prescaled", False)),
            }

        self._bit_ctx_cache = ctx
        self._bit_ctx_device = device
        return ctx

    def _build_hoisted_rotations(self, x, bit_ctx, tokens_per_expert):
        """P6-1: 对收益为正的 bit，把 gate_up 的分组旋转提到 expert 循环外做一次。

        依据：gate_up 的旋转矩阵只依赖 (group_size, seed)，与 expert 无关
        （见 triton_kernels.triton_fused_matmul_grouped_slice_rows_gf，row_start/row_end
        只切权重不进旋转）。而旋转按 K 维分组、逐行独立，因此
        `rotate(x[idx]) ≡ rotate(x)[idx]` —— 数学严格等价。

        收益判据：设 T 为本次 forward 的 token 数，某 bit 下所有 expert 的
        token 行数之和为 R。逐 expert 旋转要转 R 行，预旋转要转 T 行。
        因为 top_k > 1，主力 bit 的 R ≈ T × top_k ≫ T，收益接近 top_k 倍；
        但长尾 bit（只有几个 expert）的 R < T，预旋转反而亏，必须跳过。

        Returns:
            {bit_str: x_rot}，只包含判定为划算的 bit。
        """
        T = x.shape[0]
        cum = np.asarray(tokens_per_expert)
        counts = np.diff(np.concatenate(([0], cum)))  # 每个 expert 的 token 数

        out = {}
        for bit_str, ctx in bit_ctx.items():
            # 只有 gf 主路径（num_groups >= 2 且对齐）支持 x_is_rotated
            if not ctx["gf_gate_up"]:
                continue
            gs = ctx["gate_up_group_size"]
            in_f = ctx["gate_up_in_features"]
            if in_f % gs != 0 or in_f // gs < 2:
                continue

            rows = int(counts[self._expert_mask_by_bit[bit_str]].sum())
            if rows <= T * self.rotation_hoist_threshold:
                continue  # 预旋转不划算（长尾 bit）

            out[bit_str] = batch_rotate_input(x, gs, ctx["gate_up_seed"])

        return out

    def set_cuda_profile(self, enabled: bool = True):
        """开关 P6-0 的 CUDA event 分阶段计时。

        开启后每次 forward 会在开头和结尾各 synchronize 一次（这会串行化
        与上下层的重叠），所以**只用于测量，不要在正式跑 eval 时开**。
        结果存在 self.last_cuda_stats，可用 CudaStageProfiler.format_stats 打印。
        """
        self.enable_cuda_profile = bool(enabled)
        self._cuda_prof = CudaStageProfiler(enabled=bool(enabled))
        return self

    @torch.no_grad()
    def drop_runtime_caches(self):
        """Drop runtime caches that hold tensors on the old device; call before
        moving a layer between GPU and CPU.

        In the standby flow layer.to('cpu') only moves registered buffers,
        while the _packed dicts and gf-layout caches keep old GPU tensors
        alive (about one full copy of the packed weights per layer). After
        clearing, forward lazily rebuilds on the buffers' current device
        (buffers move together with the layer).
        """
        for w in self.bit_weights.values():
            w._gate_up_packed = None
            w._down_packed = None
        self._gf_built = False
        self._bit_ctx_cache = None

    @torch.no_grad()
    def warmup_kernels(self, seq_len: int = 2048, batch_size: int = 1):
        """预编译 Triton kernel，消除第一次 forward 的 JIT 编译冷启动开销。

        用典型形状的 dummy 输入跑一次 forward，触发所有会用到的 kernel 编译。
        因为 Triton 编译缓存是全局的（同形状同配置复用），warmup 一次后
        所有 layer 的第一次 forward 都是 warm 状态。

        Args:
            seq_len: 模拟的序列长度（默认 2048，匹配 eval 常见值）
            batch_size: 模拟的 batch size（默认 1）
        """
        device = next(self.gate.parameters()).device
        # dummy 的 dtype 必须与真实运行时输入一致（本模型是 bf16，见 qwen35_utils.py:22）。
        # 之前写死 fp16，与 bf16 的 shared_expert_gate 权重 dtype 不匹配，直接 RuntimeError。
        dtype = self.gate.weight.dtype
        dummy = torch.randn(batch_size, seq_len, self.hidden_size,
                           device=device, dtype=dtype)
        # 跑一次 forward（忽略输出），触发所有 kernel 编译
        _ = self.forward(dummy)
        torch.cuda.synchronize()

    def _gate_up_matmul(self, ctx, bit, bit_str, x, x_rot_by_bit,
                        exp_token_idx, expert_tokens, start, end, prof):
        """gate_up 方向的量化 matmul（含激活 gather）。

        抽成独立方法只为让 WxA8 子类覆盖这一处，不必复制整个 forward
        （见 quantization/wxa8/bit_partitioned_moe.py）。逻辑与抽取前逐字一致。

        Returns:
            (gate_up_out, expert_tokens) —— expert_tokens 是懒 gather 的缓存，
            可能在本次调用中才被填充，需要回传给调用方在同 expert 的后续 bit 复用。
        """
        gate_up_packed = ctx["gate_up"]
        if ctx["gf_gate_up"]:
            # P5-1: Group-First 布局，memory coalescing 更优
            # P6-1: 该 bit 若已整体预旋转，直接按 token 索引切片，kernel 内跳过旋转
            x_rot_b = x_rot_by_bit.get(bit_str)
            if x_rot_b is not None:
                with prof.stage("gather_rotated"):
                    gate_up_inp = x_rot_b[exp_token_idx]
            else:
                if expert_tokens is None:
                    with prof.stage("gather_x"):
                        expert_tokens = x[exp_token_idx]
                gate_up_inp = expert_tokens

            with prof.stage("gate_up_kernel"):
                gate_up_out = triton_fused_matmul_grouped_slice_rows_gf(
                    gate_up_inp,
                    gate_up_packed["indices_packed_gf"],
                    ctx["gate_up_codebook"],
                    gate_up_packed["norms_gf"],
                    ctx["gate_up_seed"],
                    ctx["gate_up_group_size"],
                    ctx["gate_up_in_features"],
                    2 * start, 2 * end,  # row slice
                    bit,
                    x_is_rotated=(x_rot_b is not None),
                    norms_prescaled=ctx["gate_up_norms_prescaled"],
                )
            del gate_up_inp
        else:
            if expert_tokens is None:
                with prof.stage("gather_x"):
                    expert_tokens = x[exp_token_idx]
            with prof.stage("gate_up_kernel"):
                gate_up_out = triton_fused_matmul_grouped_slice_rows(
                    expert_tokens,
                    gate_up_packed["indices_packed"],
                    ctx["gate_up_codebook"],
                    gate_up_packed["norms"],
                    ctx["gate_up_seed"],
                    ctx["gate_up_group_size"],
                    ctx["gate_up_in_features"],
                    2 * start, 2 * end,  # row slice
                    bit
                )
        return gate_up_out, expert_tokens

    def _down_matmul(self, ctx, bit, act_out, start, end, prof):
        """down 方向的量化 matmul。WxA8 子类覆盖此方法。逻辑与抽取前逐字一致。"""
        down_packed = ctx["down"]
        with prof.stage("down_kernel"):
            if ctx["gf_down"]:
                # P5-1: Group-First 布局，in_features 切片整块连续
                return triton_fused_matmul_grouped_slice_in_features_gf(
                    act_out,
                    down_packed["indices_packed_gf"],
                    ctx["down_codebook"],
                    down_packed["norms_gf"],
                    ctx["down_seed"],
                    ctx["down_group_size"],
                    start, end,  # original_start, original_end
                    ctx["down_in_features"],
                    bit,
                    norms_prescaled=ctx["down_norms_prescaled"],
                )
            return triton_fused_matmul_grouped_slice_in_features(
                act_out,
                down_packed["indices_packed"],
                ctx["down_codebook"],
                down_packed["norms"],
                ctx["down_seed"],
                ctx["down_group_size"],
                start, end,  # original_start, original_end
                ctx["down_in_features"],
                bit
            )

    @torch.no_grad()
    # ------------------------------------------------- 静态派发（graph decode）

    def _build_static_decode_state(self, x):
        """构建静态派发状态（首个符合条件的 forward 执行一次，eager 期）。

        DARTMOQ_STATIC_DISPATCH=1 显式开启；前置不满足时直接报错而非静默
        回退 host 编排（静默回退会让 capture 在别的位置死掉，反而难排查）。

        池的划分按 (bit, inter) 二元组：DartMoQ 的混合 bit 分配按 expert
        定 inter（同一 bit 内 inter 五花八门，见 2bpw checkpoint 的
        expert_offsets），而 merged kernel 的 silu 切分/n_slice 假设池内
        inter 一致 → 同 bit 按 inter 拆子池（多池 index_add_ 累加语义与
        多 bit 池相同）。down 旋转库按 bit 整体建一份共享（kernel 按
        全局 group 基序号索引，子池共用同一索引空间；逐子池建会在
        批量旋转缓存里按 num_groups 分键各存一份，白倍增显存）。
        """
        capacity = x.shape[0] * self.top_k
        self._ensure_active_bits()
        bit_ctx = self._get_bit_context(x.device)
        pools = []
        expert_bit = torch.full((self.num_experts,), -1,
                                dtype=torch.long, device=x.device)
        expert_local = torch.full((self.num_experts,), -1,
                                  dtype=torch.long, device=x.device)
        for bit_str in self.bit_weights.keys():
            by_inter = {}  # inter -> [(e, s, t)]（expert 升序，与 _active_bits 一致）
            for e in range(self.num_experts):
                for bs, _b, s, t in self._active_bits_by_expert[e]:
                    if bs == bit_str:
                        by_inter.setdefault(t - s, []).append((e, s, t))
            if not by_inter:
                continue
            ctx = bit_ctx[bit_str]
            # 该 bit 的 down 旋转库整建一份（全部子池共享；基序号空间 =
            # 该 bit packed 布局的 group 序，kernel 的 rot_base 直接索引）
            bank_groups = ctx["down"]["shape"][1] // ctx["down_group_size"]
            shared_bank = generate_batch_rotation_matrices(
                ctx["down_group_size"], ctx["down_seed"], bank_groups,
                stride=ctx["down_group_size"], device=x.device,
                dtype=torch.float16)
            for inter in sorted(by_inter):
                members = by_inter[inter]
                ps = _StaticPoolState(bit_str, members, ctx,
                                      capacity, x.dtype,
                                      cfg_getter=self._static_cfg_getter(
                                          int(bit_str)),
                                      shared_rot_bank=shared_bank)
                bi = len(pools)
                expert_bit[ps.member_ids] = bi
                expert_local[ps.member_ids] = torch.arange(
                    ps.num_members, dtype=torch.long, device=x.device)
                pools.append(ps)
                print(f"[static-dispatch] pool bit={bit_str} inter={ps.inter} "
                      f"experts={ps.num_members} gs=({ps.gs_gu},{ps.gs_dn}) "
                      f"capacity={capacity} "
                      f"blocks(gu/dn)=({ps.block_info_gu.shape[0]},"
                      f"{ps.block_info_dn.shape[0]}) "
                      f"rot_bank(shared)={ps.rot_bank.shape[0]}x{ps.gs_dn}x{ps.gs_dn}",
                      flush=True)
        if not pools:
            raise RuntimeError("static dispatch: no active bit pool found")
        self._static_decode_state = _StaticDispatchState(
            capacity, pools, expert_bit, expert_local)

    def _forward_static_decode(self, x, topk_weights, topk_indices,
                               final_hidden_states):
        """设备端定形派发的 MoE 计算。

        全部算子定形、无 host 读取：统计（index_add_）→ 池内游标（cumsum）
        → 行位次（C×C 定形比较）→ 池缓冲散写（index_copy_，含丢弃槽）→
        kernel（运行时读 expert_info 行段）→ 权重加权 scatter（index_add_）。
        padding 行（容量上界与真实行数之差）输入置零、权重为零，对合法
        token 恒等加零（bitwise 安全）。gate_up 旋转整批预提升（数学等价，
        rotate(x)[idx] ≡ rotate(x[idx])）；down 逐行从旋转库按所属 expert
        的 group 偏移选矩阵。
        """
        st = self._static_decode_state
        T = x.shape[0]
        C = st.capacity
        dev = x.device

        flat_e = topk_indices.reshape(-1).long()
        flat_w = topk_weights.reshape(-1).to(x.dtype)
        flat_t = torch.arange(T, device=dev).repeat_interleave(self.top_k)

        # 设备端统计 + 行内位次（同一 expert 的先前行数）
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=dev)
        counts.index_add_(0, flat_e, st.ones_c)
        same = flat_e.unsqueeze(0) == flat_e.unsqueeze(1)
        prior = same.long().tril(-1).sum(1)
        pool_of_row = st.expert_bit.index_select(0, flat_e)
        local_of_row = st.expert_local.index_select(0, flat_e)

        for bi, ps in enumerate(st.pools):
            m_counts = counts.index_select(0, ps.member_ids)
            cursors = m_counts.cumsum(0, dtype=torch.int32)
            cursors = cursors - m_counts  # 排他前缀和 = 池内行游标
            ps.expert_info_gu[:, 0].copy_(cursors)
            ps.expert_info_gu[:, 1].copy_(m_counts)
            ps.expert_info_dn[:, 0].copy_(cursors)
            ps.expert_info_dn[:, 1].copy_(m_counts)

            pool_total = m_counts.sum()
            in_pool = pool_of_row == bi
            # 非本池行的局部索引指向别的池（合法于彼池、可能越界于本池），
            # clamp 到本池范围防 gather 越界；其行游标本就不参与 dst
            # （torch.where 丢弃），仅保证索引合法
            row_cursor = cursors.index_select(
                0, local_of_row.clamp(0, ps.num_members - 1)).to(torch.long)
            dst = torch.where(in_pool, row_cursor + prior, C)  # 丢弃槽 = C

            ps.tokens_buf.zero_()
            ps.tokens_buf.index_copy_(0, dst, flat_t)
            ps.weights_buf.zero_()
            ps.weights_buf.index_copy_(0, dst, flat_w)
            ps.slot_local_buf.zero_()
            ps.slot_local_buf.index_copy_(0, dst, local_of_row)
            tokens_c = ps.tokens_buf[:C]
            weights_c = ps.weights_buf[:C]
            mask_rows = st.arange_c < pool_total

            self._static_pool_compute(
                ps, x, tokens_c, weights_c, mask_rows, final_hidden_states)

        return final_hidden_states

    # 各模式子类声明：本类的 _static_pool_compute 已按自身 kernel 语义实现
    # （未声明的子类被 forward 的静态派发钩子挡掉，防把基类数值烤进图）。
    # 本次移植只接通 int8（wxa8 子类声明 True）；基类 A16 的静态池 kernel
    # （triton_kernels 的 merged/static 变体）未移植，保持 False
    _STATIC_DISPATCH_READY = False

    @staticmethod
    def _static_cfg_getter(bit):
        """静态池的 tile 选档函数（本模式 kernel 的 config 入口）。"""
        def getter(direction, B):
            return _get_fused_grouped_config(bit, direction=direction, B=B)
        return getter

    @staticmethod
    def _static_scatter_tail(dn, tokens_c, weights_c, mask_rows,
                             final_hidden_states):
        """静态池 scatter 尾（各模式共用）：padding 行显式清零后加权累加。

        dn 的 padding 行（行号 ≥ pool_total）不属于任何成员的行段，GEMM
        kernel 的 store 掩码（rb < token_start + token_count）从不写入它们，
        内容是 torch.empty 的分配器残留；weights_c 在 padding 行虽为 0，但
        0 × inf/NaN = NaN——残留非有限时经 index_add_ 把 token 0 污染成
        NaN（多池拆分必有 padding 行故显形；单池 pool_total=C 无 padding，
        同一缺陷不触发。残留是否非有限取决于 prefill 的分配历史 → 呈实例
        依赖；host 路径无 padding 行故从不触发）。
        masked_fill_ 按掩码覆写（非算术），对任意残留安全；定形定形状、
        设备端掩码，capture 合法。开销 = 每池每层一个小 fill 节点。
        """
        dn.masked_fill_(~mask_rows.unsqueeze(1), 0)
        dn.mul_(weights_c.unsqueeze(1))
        final_hidden_states.index_add_(0, tokens_c, dn)

    def forward(self, hidden_states):
        """
        前向推理：按 expert 处理，对每个 bit 分别反量化 + GEMM。
        """
        prof = self._cuda_prof
        prof.begin_round()

        t0 = time.time()

        batch_size, seq_len, hidden_dim = hidden_states.shape
        x = hidden_states.reshape(-1, hidden_dim)

        final_hidden_states = torch.zeros_like(x)

        # 懒初始化 CPU 端 expert_offsets（避免每轮 .item() D2H 同步）
        # 兼容 from_build_block 和手动构造两种路径
        if not self._expert_offsets_cpu:
            for bit_str, offsets in self.expert_offsets.items():
                self._expert_offsets_cpu[bit_str] = offsets.cpu().tolist()

        # P5-1: 懒构建 group-first 布局（首次 forward 时一次性转换）
        # 不修改 state_dict 格式，纯运行时优化，内存占用与原布局相同（只是排列不同）
        # 注意用 getattr 而非 hasattr：drop_runtime_caches 会把 _gf_built 重置为
        # False（跨设备搬移后需按新设备重建），hasattr 恒真会跳过重建
        if not getattr(self, '_gf_built', False):
            self._gf_built = True
            for w in self.bit_weights.values():
                # 先触发 property 重建（from_metadata 路径下 _packed 为 None）
                _ = w.gate_up_packed
                _ = w.down_packed
                w._build_group_first("both")
            # gf 布局重建后 packed 字典内容变了，bit context 缓存需失效
            self._bit_ctx_cache = None

        # P6-2: 预计算每 expert 的活跃 bit 列表（依赖上面的 _expert_offsets_cpu）
        self._ensure_active_bits()

        t1 = time.time()

        # shared expert (behavioral-convention dispatch, see
        # quantization/common/adapters.py)
        t_shared_start = time.time()
        if self.shared_mode != "none":
            with prof.stage("shared_expert"):
                shared_add = shared_forward(self, x, self.shared_mode)
                if shared_add is not None:
                    final_hidden_states.add_(shared_add)
                    del shared_add
        t_shared_end = time.time()
        t2 = t_shared_end

        # router (behavioral-convention dispatch; deepseek-family gates
        # require 3D input, hence hidden_states is passed)
        t_router_start = time.time()
        with prof.stage("router"):
            topk_weights, topk_indices, router_logits = router_forward(
                self.gate, hidden_states, x, self.router_mode, self.top_k)
        t_router_end = time.time()
        t3 = t_router_end

        # ------------------------------------------------ 静态派发（graph decode）
        # DARTMOQ_STATIC_DISPATCH=1 且子类已实现 _static_pool_compute 时，
        # decode 形状（N*top_k ≤ 上界）走设备端定形派发，替代下面的 host
        # 编排（DtoH 统计 → pool 字典 → 变长 cat → H2D 表）。构建仍需
        # host 逻辑，只在 eager 期执行一次（capture 之前）。
        if (os.environ.get(_STATIC_DISPATCH_ENV, "0") == "1"
                and type(self)._STATIC_DISPATCH_READY):
            _st = getattr(self, "_static_decode_state", None)
            _rows = x.shape[0] * self.top_k
            if ((_st is None or _rows > _st.capacity)
                    and _rows <= _STATIC_DISPATCH_MAX_CAPACITY):
                self._build_static_decode_state(x)
                _st = self._static_decode_state
            if _st is not None and _rows <= _st.capacity:
                out = self._forward_static_decode(
                    x, topk_weights, topk_indices, final_hidden_states)
                return out.reshape(batch_size, seq_len, hidden_dim)

        # 优化结构：expert -> bit
        with prof.stage("sort_prep"):
            # Flatten: (N, top_k) -> (N * top_k,)
            flat_expert_indices = topk_indices.flatten()
            flat_expert_weights = topk_weights.flatten()
            flat_token_indices = torch.arange(x.shape[0], device=x.device).repeat_interleave(self.top_k)

            # 按 expert 排序
            idxs = flat_expert_indices.argsort()
            sorted_experts = flat_expert_indices[idxs]
            sorted_weights = flat_expert_weights[idxs]
            sorted_tokens = flat_token_indices[idxs]

            # 统计每个 expert 的 token 数
            tokens_per_expert = sorted_experts.bincount(minlength=self.num_experts).cpu().numpy().cumsum(0)

        # 对每个 expert 处理
        t_compute_start = time.time()

        # 统计各阶段时间
        time_dequant_total = 0.0
        time_gemm_total = 0.0
        time_triton_total = 0.0
        active_experts_count = 0
        active_bits_count = 0

        # ---- P6-2: 每 bit 的常量准备（每层每 bit 做一次，不进 expert 循环）----
        # 原先 codebook.to(device) / 字典查找 每层要重复约 512 次。
        bit_ctx = self._get_bit_context(x.device)

        # ---- P6-1: gate_up 旋转预提升 ----
        # gate_up 的旋转矩阵只依赖 (group_size, seed)，与 expert 无关；
        # 而 top_k 导致同一 token 在 per-expert 循环里被重复旋转 top_k 次。
        # 这里对"该 bit 下总行数 > T * 阈值"的 bit 预先整体旋转一次，
        # 循环内直接按 token 索引切片。数学严格等价（旋转逐行独立）。
        x_rot_by_bit = {}
        if self.enable_rotation_hoist:
            with prof.stage("rotation_hoisted"):
                x_rot_by_bit = self._build_hoisted_rotations(
                    x, bit_ctx, tokens_per_expert,
                )

        for expert_idx in range(self.num_experts):
            # print(f"  [DEBUG] expert_idx: {expert_idx}", flush=True)
            # t0 = time.time()
            end_idx = tokens_per_expert[expert_idx]
            start_idx = 0 if expert_idx == 0 else tokens_per_expert[expert_idx - 1]

            if start_idx == end_idx:
                continue

            active_experts_count += 1

            # 取当前 expert 的所有 token
            exp_token_idx = sorted_tokens[start_idx:end_idx]
            expert_weights = sorted_weights[start_idx:end_idx].unsqueeze(1)

            # P6-2: expert_tokens 改为懒 gather —— 只有存在"未预旋转"的 bit 时才需要原始 x 切片。
            # 全部 bit 都走预旋转时（本模型的主力场景），这次 gather 完全省掉。
            expert_tokens = None
            # P6-2: 去掉 torch.zeros_like(expert_tokens)（每层 256 次 8MB 分配）。
            # 每个 expert 实际只有一个活跃 bit，首个 bit 的结果直接接管，后续 bit 才累加。
            expert_out = None

            # P6-2: 只遍历该 expert 真正活跃的 bit（预计算），
            # 原先每层空转约 512 次 `actual_inter_size == 0` 的 continue。
            for bit_str, bit, start, end in self._active_bits_by_expert[expert_idx]:
                t_triton_start = time.time()
                active_bits_count += 1

                actual_inter_size = end - start
                ctx = bit_ctx[bit_str]

                # ========== 量化 matmul：gate_up → silu*up → down ==========
                # 两个 matmul 抽成方法，WxA8 子类只覆盖这两处
                gate_up_out, expert_tokens = self._gate_up_matmul(
                    ctx, bit, bit_str, x, x_rot_by_bit,
                    exp_token_idx, expert_tokens, start, end, prof,
                )

                with prof.stage("silu_mul"):
                    gate_out = gate_up_out[:, :actual_inter_size]
                    up_out = gate_up_out[:, actual_inter_size:]
                    del gate_up_out

                    act_out = F.silu(gate_out) * up_out
                    del gate_out, up_out

                down_out = self._down_matmul(ctx, bit, act_out, start, end, prof)
                del act_out

                # P6-2: 首个活跃 bit 直接接管输出，避免 zeros_like 预分配
                if expert_out is None:
                    expert_out = down_out
                else:
                    expert_out += down_out
                    del down_out
                t_triton_end = time.time()
                time_triton_total += t_triton_end - t_triton_start
                # print(f"    [DEBUG] bit: {bit}, time: {t_triton_end - t_triton_start:.4f}s", flush=True)
                # ==========================================

            # 该 expert 无任何活跃 bit（理论上不会发生，防御性跳过）
            if expert_out is None:
                del expert_weights, exp_token_idx
                continue

            # 累加回最终结果
            # 用 index_add_ 替代 scatter_reduce_(sum)：
            #   - 无需把 index 从 (M,) expand/repeat 到 (M, H)，省一次大张量分配+拷贝
            #   - 语义完全等价（按行累加，dim=0）
            with prof.stage("scatter"):
                expert_out.mul_(expert_weights)
                final_hidden_states.index_add_(0, exp_token_idx, expert_out)

            del expert_out, expert_tokens, expert_weights, exp_token_idx
            # t1 = time.time()
            # print(f"    [DEBUG] expert_idx: {expert_idx}, time: {t1 - t0:.4f}s", flush=True)

        # Cleanup
        del flat_expert_indices, flat_expert_weights, flat_token_indices
        del idxs, sorted_experts, sorted_weights, sorted_tokens, tokens_per_expert
        # P6-1: 及时释放预旋转缓存（每 bit 一块 (T, hidden) fp16，本模型约 256MB）
        x_rot_by_bit.clear()
        del x_rot_by_bit

        t_compute_end = time.time()
        t4 = t_compute_end

        result = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)

        # P6-0: 取本轮 CUDA event 统计（这里做本 forward 唯一的一次 synchronize）
        if self.enable_cuda_profile:
            self.last_cuda_stats = prof.end_round()

        if self.enable_timing:
            t5 = time.time()
            self.last_timings = {
                'total': t5 - t0,
                'init': t1 - t0,
                'shared': t_shared_end - t_shared_start,
                'router': t_router_end - t_router_start,
                'sort_scatter_prep': t_compute_start - t_router_end,
                'compute': t_compute_end - t_compute_start,
                'triton': time_triton_total,
                'dequant': time_dequant_total,
                'gemm': time_gemm_total,
                'cleanup_reshape': t5 - t_compute_end,
                'active_experts': active_experts_count,
                'active_bits': active_bits_count,
            }

            # 打印详细时间（仅第一次 forward）
            if not hasattr(self, '_log_printed'):
                self._log_printed = True
                lt = self.last_timings
                print(f"  [WxA16BitPartitionedGroupMoE] forward total: {lt['total']:.4f}s", flush=True)
                print(f"    init: {lt['init']:.4f}s, shared: {lt['shared']:.4f}s, router: {lt['router']:.4f}s", flush=True)
                print(f"    compute: {lt['compute']:.4f}s (triton: {lt['triton']:.4f}s, dequant: {lt['dequant']:.4f}s, gemm: {lt['gemm']:.4f}s)", flush=True)
                print(f"    reshape+cleanup: {lt['cleanup_reshape']:.4f}s, active_experts: {lt['active_experts']}, active_bits: {lt['active_bits']}", flush=True)
                # P6-0: wall-clock 拆分不可信（无 sync，异步下会把耗时记错段），
                # 开了 CUDA profile 时以下面这份 GPU 时间为准。
                if self.enable_cuda_profile:
                    print(CudaStageProfiler.format_stats(self.last_cuda_stats), flush=True)

        # olmoe-style DecoderLayers expect a (hidden, router_logits) 2-tuple
        # (router_logits is unused for aux loss at eval time; it only needs
        # to be returned so the layer can unpack)
        if self.mlp_returns_tuple:
            return result, router_logits
        return result
