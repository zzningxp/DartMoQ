"""TurboQuant Model — Near-optimal weight quantization with on-the-fly dequantization.

TurboQuant 是一种近最优的权重量化方法，结合随机正交旋转和 Lloyd-Max 标量量化，
实现高效的在线反量化，无需存储完整量化权重，显著降低内存占用。

论文来源: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
(Zandieh et al., 2025, arXiv:2504.19874)

核心特性:
- 单遍量化：一次遍历即可完成量化，无需迭代微调
- 在线反量化：推理时实时计算，节省 4-8 倍内存
- 支持多种量化策略：基础量化、残差量化、哈希表压缩
- 支持 Transformer 模型的无缝集成

基础使用示例:
    from turboquant_model import TurboQuantConfig, quantize_model

    # 创建量化配置
    config = TurboQuantConfig(bit_width=4, seed=42)
    # 量化模型（返回FP32近似用于评估）
    model = quantize_model(model, config)
    # 启用融合模式（真正的4-bit存储和在线反量化）
    enable_fused_mode(model)

模块结构:
- codebook: Lloyd-Max 最优码本计算
- rotation: 随机正交旋转矩阵生成
- quantize: 核心 TurboQuant 量化实现
- residual: 残差量化扩展
- module: 量化线性层和嵌入层
- model: 模型级量化接口
- norm_compression: 范数压缩算法
- entropy_codec: 熵编码压缩
- norm_calibration: 范数校准
- hash_table: 哈希表压缩
- dartmoq_backend: DartMoQ 集成用 fake-quant 工具
"""

import sys

# The upstream TurboQuant sources import their package as ``turboquant_model``.
# This repository vendors the same code under ``turboquant_utils``; expose a
# local alias so those absolute imports resolve without installing/renaming it.
sys.modules.setdefault("turboquant_model", sys.modules[__name__])

# ------------------------------
# 核心模块导入
# ------------------------------

# 码本模块：Lloyd-Max 最优码本生成
from .codebook import get_codebook

# 旋转模块：随机正交旋转矩阵（QR分解/Hadamard变换）
from .rotation import generate_rotation_matrix

# 量化核心模块：基础 TurboQuant 量化算法
from .quantize import (
    pack_4bit,               # 4-bit索引打包
    unpack_4bit,             # 4-bit索引解包
    turboquant_quantize,     # 量化并返回FP32近似（模拟模式）
    turboquant_quantize_packed,  # 量化并返回打包格式（存储模式）
)

# 残差量化模块：多级残差量化扩展
from .residual import (
    residual_quantize,               # 单级残差量化（模拟）
    residual_quantize_packed,        # 单级残差量化（打包）
    multi_residual_quantize,         # 多级残差量化（模拟）
    multi_residual_quantize_packed,  # 多级残差量化（打包）
    alternating_residual_quantize,        # 交替残差量化（模拟）
    alternating_residual_quantize_packed, # 交替残差量化（打包）
    merge_residual_passes,           # 合并多个残差通道
    merge_and_requantize,            # 合并并重量化
)

# 模块层：量化线性层和嵌入层实现
from .module import (
    TurboQuantLinear,    # TurboQuant 量化线性层
    SharedScratchPool,   # 共享 scratch 内存池
    QuantizedEmbedding,  # 量化嵌入层
)

# 模型级接口：完整模型量化和序列化
from .model import (
    TurboQuantConfig,      # 量化配置类
    quantize_model,        # 标准模型量化（推荐）
    quantize_model_advanced,  # 高级模型量化（自定义配置）
    save_quantized,        # 保存量化模型
    load_quantized,        # 加载量化模型
    enable_prefetch_chain, # 启用预取链优化
    disable_prefetch_chain, # 禁用预取链优化
)

# 范数压缩模块：高效存储归一化因子
from .norm_compression import (
    FactoredNorms,         # 因子化范数存储类
    factorize_norms,       # 范数因子化
    reconstruct_norms,     # 范数重构
    norm_bpw,              # 计算范数的比特率
)

# 熵编码模块：无损压缩量化索引
from .entropy_codec import (
    compress_indices,      # 压缩索引
    decompress_indices,    # 解压索引
    compute_entropy,       # 计算索引熵
    measure_compressed_bpw, # 测量压缩后的比特率
)

# 范数校准模块：推理时校准范数
from .norm_calibration import (
    calibrate_norms,              # 范数校准
    calibrate_norms_blockwise,     # 分块范数校准
    CalibrationConfig,             # 校准配置类
    collect_calibration_data,      # 收集校准数据
)

# 哈希表压缩模块：基于哈希的权重压缩
from .hash_table import (
    HashTableConfig,       # 哈希表配置类
    HashWeightTable,       # 哈希权重表类
    compute_group_stats,   # 计算分组统计信息
    quantize_stats,        # 量化统计信息
    build_hash_keys,       # 构建哈希键
    multi_head_lookup,     # 多头查找
    train_hash_table,      # 训练哈希表
    hash_compress,         # 哈希压缩
    hash_decompress,       # 哈希解压
    compute_bpw,           # 计算比特率
)

# DartMoQ 集成工具：只使用本项目内的 TurboQuant 源码
from .dartmoq_backend import (
    TurboFakeQuantResult,
    get_linear_bit_from_dartmoq_quantizer,
    is_turbo_fake_quant_supported,
    quantize_linear_if_turbo_supported,
    turbo_fake_quant_linear,
)

# 版本号
__version__ = "0.1.0"

# ------------------------------
# 公开 API 导出列表
# 按功能模块组织，便于用户查找和使用
# ------------------------------
__all__ = [
    # ============ 码本模块 ============
    "get_codebook",

    # ============ 旋转模块 ============
    "generate_rotation_matrix",

    # ============ 量化核心 ============
    "pack_4bit",
    "unpack_4bit",
    "turboquant_quantize",
    "turboquant_quantize_packed",

    # ============ 残差量化 ============
    "residual_quantize",
    "residual_quantize_packed",
    "multi_residual_quantize",
    "multi_residual_quantize_packed",
    "alternating_residual_quantize",
    "alternating_residual_quantize_packed",
    "merge_residual_passes",
    "merge_and_requantize",

    # ============ 量化模块 ============
    "TurboQuantLinear",
    "SharedScratchPool",

    # ============ 模型接口 ============
    "TurboQuantConfig",
    "quantize_model",
    "quantize_model_advanced",
    "save_quantized",
    "load_quantized",
    "enable_prefetch_chain",
    "disable_prefetch_chain",

    # ============ 范数压缩 ============
    "FactoredNorms",
    "factorize_norms",
    "reconstruct_norms",
    "norm_bpw",

    # ============ 熵编码 ============
    "compress_indices",
    "decompress_indices",
    "compute_entropy",
    "measure_compressed_bpw",

    # ============ 范数校准 ============
    "calibrate_norms",
    "CalibrationConfig",
    "collect_calibration_data",

    # ============ 哈希表压缩 ============
    "HashTableConfig",
    "HashWeightTable",
    "compute_group_stats",
    "quantize_stats",
    "build_hash_keys",
    "multi_head_lookup",
    "train_hash_table",
    "hash_compress",
    "hash_decompress",
    "compute_bpw",

    # ============ DartMoQ 集成工具 ============
    "TurboFakeQuantResult",
    "get_linear_bit_from_dartmoq_quantizer",
    "is_turbo_fake_quant_supported",
    "quantize_linear_if_turbo_supported",
    "turbo_fake_quant_linear",
]
