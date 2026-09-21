"""WxA8 量化推理实现（INT8 激活 + 混合 bit 权重 + INT8 Tensor Core）。

存储格式与 WxA16 完全相同，checkpoint 通用；只有 forward 的 matmul 不同。
"""

from .bit_partitioned_moe import WxA8BitPartitionedGroupMoE
from .linear import WxA8Linear

__all__ = ["WxA8BitPartitionedGroupMoE", "WxA8Linear"]
