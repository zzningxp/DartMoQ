# E7 checkpoint 存储口径（a8s8m1/m2 十格，全 metadata 计入）

生成：2026-11-01 00:48，脚本 `test/test_e7_ckpt_storage.py`（纯 CPU，只读
safetensors 头 + meta.json，不加载张量）：

```
conda run -n dart312 python test/test_e7_ckpt_storage.py
```

数据源：`quant_ckpt/*-s1` 十个 checkpoint（save_quantized_model 落盘）。
参数量反推复用 `bench_throughput._orig_params_from_meta`（meta.json 记录的
量化前形状 + 0-bit 专家数反推）+ 文件内 fp16 残余张量元素数；dsv2-2bpw
交叉验证 15.71 B 与 bench 实测分毫不差，eff bpw 2.8503 vs bench 2.850。

## key 分类（与 bench_throughput.ckpt_report 一致，metadata 细分）

- `raw_moe`：`mlp.bit_weights.{b}.*_indices_packed`（MoE 打包权重位）
- `raw_attn`：attn/shared 的 `*.packed_indices`（wxa16 8-bit 打包）
- `raw_fp16`：未量化参数（embed_tokens / lm_head / router / layernorm）
- metadata = `*codebook` + `*norms` + `qmeta/*`（含量化 seed 标量）
- **0-bit 专家在文件里完全不存在**（非 mask，零字节）——统一建模
  "剪枝单元零 metadata 开销" 的直接证据

## 两种口径

- **as-stored**：落盘实值（norms 是 **F16**）
- **fp32-refold**：norms 按 4 B/元素折算（paper.txt "one FP32 norm per
  group" 口径）

## 全表

| ckpt | params B | 落盘 GiB | eff bpw | meta bpw | FP32折算 GiB | eff bpw | meta bpw |
|---|---|---|---|---|---|---|---|
| dsv1-1bpw  | 16.38 | 3.764 | 1.9745 | 0.0859 | 3.928 | 2.0604 | 0.1717 |
| dsv1-2bpw  | 16.38 | 5.451 | 2.8594 | 0.1139 | 5.668 | 2.9732 | 0.2277 |
| dsv2-1bpw  | 15.71 | 3.583 | 1.9597 | 0.0837 | 3.736 | 2.0433 | 0.1673 |
| dsv2-2bpw  | 15.71 | 5.212 | 2.8503 | 0.1096 | 5.412 | 2.9598 | 0.2191 |
| moon-1bpw  | 15.96 | 4.059 | 2.1845 | 0.0797 | 4.207 | 2.2642 | 0.1593 |
| moon-2bpw  | 15.96 | 5.701 | 3.0680 | 0.0981 | 5.883 | 3.1661 | 0.1961 |
| olmoe-1bpw | 6.92  | 1.513 | 1.8785 | 0.0872 | 1.583 | 1.9656 | 0.1743 |
| olmoe-2bpw | 6.92  | 2.243 | 2.7846 | 0.1122 | 2.333 | 2.8967 | 0.2243 |
| qwen3-1bpw | 30.53 | 6.025 | 1.6950 | 0.0696 | 6.272 | 1.7646 | 0.1391 |
| qwen3-2bpw | 30.53 | 9.335 | 2.6263 | 0.0878 | 9.647 | 2.7141 | 0.1756 |

## 口径反推（dsv2 a8s8m2 = 5.415 GiB / 2.9614 bpw / meta 0.252 bpw）

- 5.415 GiB / 2.9614 bpw ✅ 复现为 **FP32-refold 口径**（5.412 / 2.9598，
  差 0.05%，来源 aux 文件或舍入）：norms 按 FP32 折算、分母全参数 15.71 B。
- **metadata 0.252 bpw 复现不出来**：文件实测 FP32 折算 0.2191。0.252 是
  理论值（README「TurboQuant ~0.252 bpw overhead」：g=128 → FP32 norm
  0.25 + codebook ≈ 0.002）。实测 −13% 的原因 = 0-bit 剪枝单元不带
  norms/codebook——这个折扣本身可作统一建模优点的定量证据。

## 待定（需要作者决定）

paper.txt 写 "FP32 norm"、落盘是 F16，两处口径不一致。二选一：
按落盘 F16 报 as-stored 列，或把实现改成真存 FP32。

## 备注

- moon-2bpw eff 3.07/3.17 偏高（分配后 raw bpw 高于名义 2），写作时注意
  用「平均有效 bpw」表述，不要与名义 target 混同。
- 内部编号（E7 等）不得出现在 rebuttal 文本，只在本目录内部文档使用。
