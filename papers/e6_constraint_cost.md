# E6 约束代价（dp vs 无约束 MCKP oracle，同预算同损失表）

生成：2026-11-01 ~03:00，PPL 数据源 logs/1031.e4b{,2_bench,3}.log（E4b
unconstrained 格，seed=42、extrapolate@1.0）；proxy 侧 125 格全量见
logs/e4_correlation_summary.txt [5]。汇总脚本 test/test_e4_monotone.py。

## 五模型终表（extrapolate @1.0bpw）

| 模型 | proxy gap¹ | dp wiki/c4 | unc wiki/c4 | PPL gap wiki / c4 |
|---|---|---|---|---|
| qwen3-30b-a3b | −1.72% | 10.95 / 18.73 | 10.87 / 18.59 | **−0.8% / −0.7%** |
| moonlight | −1.15% | 15.93 / 37.58 | 16.42 / 40.89 | +3.0% / +8.8% |
| deepseek-v2-lite | −1.01% | 9.18 / 18.96 | 9.34 / 19.35 | +1.7% / +2.1% |
| olmoe-7b-1b | −0.78% | 22.42 / 40.05 | 23.57 / 41.86 | +5.1% / +4.5% |
| deepseek-v1-moe-16b | −0.66% | 10.41 / 21.25 | 10.71 / 21.93 | +2.9% / +3.2% |

¹ proxy gap = unc/dp − 1（负 = 松弛最优更低，逻辑必然，125 格 max +0.00%）

proxy 侧全量（125 格）：mean −2.43% / median −0.68% / max +0.00%。

同场对照（dp vs uniform/random，10/10 全胜，量级 3.7×–486×）：
olmoe uniform 10899/4984 random 363.5/328.6；dsv1 uniform 522.9/562.1
random 82.2/116.7；dsv2 uniform 29.0/39.8 random 21.6/30.6；moon uniform
142.8/173.5 random 111.0/185.2；qwen3 uniform 1108.7/938.0 random
2427.7/1843.0。

## 解读（rebuttal 措辞基础）

- **proxy 侧**：单调约束代价 0.66%–1.72%（@1.0 五模型），逻辑上限为 0。
- **PPL 侧**：4/5 模型 unc 反而略差（+1.7%–5.1% wiki），唯一例外 qwen3
  （−0.8%）恰是 proxy gap 最大的模型（−1.72%）——proxy 差距最大的地方
  松弛才换来一点 PPL 收益。
- **不要声称跨模型剂量-反应**：五点 (proxy gap, PPL gap) 的 spearman
  仅 ~0.15，无一致映射。安全口径：约束代价 ≤1.7%（proxy 上界、逻辑
  必然）；PPL 侧无一致代价，全部差异在近平局量级（<5% proxy 差距档
  的 bucket 一致率 56%，正是 proxy 分辨率以下），远小于 uniform/random
  的 3.7×–486× 差距。
- 主算法保留单调约束的理由：packed 部署形态（同 bit 连续排布）、解的
  规整性；MCKP 0.12 s/层 vs DP ~18 s/层 都是标定期的离线开销。

## W4 覆盖范围备忘

本表只覆盖「单调 bitwidth 约束」一轴；排序交叉（ranking crossovers）
与块粒度（vs per-neuron）两维不给 gap 数字，只给论证（近平界分析 +
块是 packed 部署最小单元）。W4 原文无 "oracle" 字眼，要的是 gap 量化。
