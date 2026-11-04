# 推理速度/显存 rebuttal 数据（1031 bench + graph decode）

- 目的：补齐评审要求的口径 —— checkpoint GB、含 metadata/mask 的 effective bpw、
  峰值推理显存、prefill/decode 延迟、tokens/s、不同 batch/seq 吞吐；并回应
  "MoE 只访问 routed experts、混合精度解码、不规则调度、负载不均可能抵消带宽节省"
- 环境：RTX 5090 ×2（32GiB each），torch 2.9.0+cu128，triton 3.5.0，
  transformers 5.20.1，commit `5d6b6e2+`
- 日志/结果：
  - bench：`logs/1104.1.full.bench.log`（FRESH 全量 20 格，当前基准）+
    `results/throughput_*.json`；此前分轮 `logs/1031.e4b2_bench.log`（8 格）、
    `logs/1102.1.bench.log`（三轮修复，见 roadmap 阶段 5）。marker
    `logs/bench_done/`，入口 `run.bench.sh`（`FRESH=1` 清 marker 全量重跑）；
    `logs_parser.py` 自动识别 bench 日志出表（csv/json/md）
  - graph：`logs/1031.graph_qwen3.log` +
    `results/graph_decode_qwen3-30b-a3b_wxa8_5d6b6e2+_261031_233129.json`，
    入口 `run.graph.sh`（`graph_decode.py`）
  - profile：`run.profile.sh`（`profile_graph_decode.py`）→
    `results/profile_graph_{qwen3-30b-a3b,deepseek-v2-lite}_wxa8_1101_*.json`
- 状态（2026-11-04）：bench **20/20 格全 PASS**（1104 全量重跑，一致性检查
  全过）；graph decode 见下表；graph 逐 kernel profile 完成，调优主线定为
  kernel 数量（见下）；fork 55 tps 口径已对账关闭

## bench 全量 20 格（1104.1 FRESH 重跑，全 PASS）

五模型 × {1,2}bpw × {wxa16,wxa8}；每行内 `wxa16 / wxa8` 双列。pf=2048×4
tok/s，dec B1/B4=128 tok wall tok/s（B4 为四流合计，每序列 ÷4），
TTFT 为 s。

| ckpt | eff bpw | ckpt GB（MoE / Attn+Sh / fp16 残余） | 加载后 VRAM | 峰值 VRAM | pf 2048×4 | dec B1 | dec B4 | TTFT |
|---|---|---|---|---|---|---|---|---|
| dsv2-1bpw | 1.960 | 3.58（1.95 / 0.84 / 0.79） | 3.90 | 4.79 / 4.77 | 16956 / 25891 | 23.31 / 23.70 | 54.9 / 58.6 | 0.22 / 0.20 |
| dsv2-2bpw | 2.850 | 5.21（3.58 / 0.84 / 0.79） | 5.72 | 6.63 / 6.62 | 14338 / 22242 | 21.07 / 21.86 | 46.2 / 49.1 | 0.27 / 0.24 |
| qwen3-1bpw | 1.695 | 6.03（3.98 / 0.86 / 1.18） | 6.50 | 7.97 / 7.89 | 9245 / 13774 | 12.17 / 12.69 | 26.1 / 28.9 | 0.49 / 0.43 |
| qwen3-2bpw | 2.626 | 9.34（7.29 / 0.86 / 1.18） | 10.07 | 11.55 / 11.48 | 8110 / 12101 | 11.73 / 12.23 | 24.8 / 27.7 | 0.55 / 0.49 |
| olmoe-1bpw | 1.878 | 1.51（0.87 / 0.25 / 0.39） | 1.64 | 3.02 / 2.98 | 35538 / 53166 | 41.04 / 42.53 | 89.1 / 98.2 | 0.13 / 0.12 |
| olmoe-2bpw | 2.785 | 2.24（1.60 / 0.25 / 0.39） | 2.41 | 3.81 / 3.76 | 29162 / 44660 | 35.66 / 37.09 | 71.0 / 77.4 | 0.16 / 0.14 |
| dsv1-1bpw | 1.975 | 3.76（2.03 / 0.95 / 0.79） | 4.23 | 6.54 / 6.48 | 16190 / 25975 | 22.49 / 22.96 | 53.2 / 57.5 | 0.23 / 0.21 |
| dsv1-2bpw | 2.859 | 5.45（3.71 / 0.95 / 0.79） | 6.07 | 8.40 / 8.34 | 13583 / 21743 | 20.49 / 21.25 | 44.4 / 47.8 | 0.29 / 0.25 |
| moon-1bpw | 2.185 | 4.06（1.96 / 0.84 / 1.26） | 4.36 | 5.56 / 5.22 | 16581 / 26177 | 21.29 / 22.25 | 53.0 / 58.1 | 0.20 / 0.18 |
| moon-2bpw | 3.068 | 5.70（3.60 / 0.84 / 1.26） | 6.13 | 7.34 / 7.00 | 14151 / 22726 | 20.79 / 21.01 | 48.8 / 52.8 | 0.23 / 0.21 |

（decode 为 eager host 编排路径；CUDA graph 数字见下节。1031 首轮 8 格的
分项详表保留如下，数值与 1104 复测一致。）

## 1031 首轮 8 格分项详表

### 存储 / 显存

| ckpt | checkpoint GB（MoE / Attn+Sh / fp16 残余） | eff bpw | 加载后 VRAM | 峰值 VRAM (wxa16/wxa8) |
|---|---|---|---|---|
| dsv2-1bpw | 3.58（1.95 / 0.84 / 0.79） | 1.960 | 3.90 | 4.79 / 4.77 |
| dsv2-2bpw | 5.21（3.58 / 0.84 / 0.79） | 2.850 | 5.72 | 6.63 / 6.62 |
| qwen3-1bpw | 6.03（3.98 / 0.86 / 1.18） | 1.695 | 6.50 | 7.97 / 7.89 |
| qwen3-2bpw | 9.34（7.29 / 0.86 / 1.18） | 2.626 | 10.07 | 11.55 / 11.48 |

### prefill 吞吐（tok/s，wxa16 / wxa8）

| ckpt | L128 B1 | L2048 B1 | L2048 B4 |
|---|---|---|---|
| dsv2-1bpw | 762 / 850 | 9545 / 12004 | 16882 / 25874 |
| dsv2-2bpw | 596 / 680 | 7869 / 9631 | 14299 / 22187 |
| qwen3-1bpw | 327 / 375 | 4008 / 4786 | 9223 / 13823 |
| qwen3-2bpw | 272 / 319 | 3251 / 3930 | 8095 / 12019 |

wxa8 长序列 prefill 比 wxa16 快 26–40%（B4 档放大最明显）。

### decode（B1；wxa16 / wxa8）

| ckpt | step mean/p99 (ms) | tok/s | tok/s B4（每序列） | TTFT (s) |
|---|---|---|---|---|
| dsv2-1bpw | 42.9/47.5 · 41.8/46.0 | 23.3 / 23.9 | 53.5 / 57.4（13.4/14.4） | 0.22 |
| dsv2-2bpw | 47.6/50.2 · 46.2/48.6 | 21.0 / 21.7 | 46.0 / 48.9（11.5/12.2） | 0.27 |
| qwen3-1bpw | 81.9/85.5 · 78.8/83.5 | 12.2 / 12.7 | 26.6 / 28.8（6.6/7.2） | 0.49 |
| qwen3-2bpw | 87.0/92.6 · 82.6/85.1 | 11.5 / 12.1 | 24.4 / 27.5（6.1/6.9） | 0.57 |

### MoE 阶段分解（每 decode step 全层平均；回应"调度抵消带宽节省"）

- dsv2-1bpw wxa16：MoE 全阶段 37.3ms / step 42.9ms（87%）；compute 27.1
  （triton 22.3）、sort/scatter 3.2、shared 5.4、router 1.3 → 调度开销
  （sort/scatter+router）仅占 step ~10%
- qwen3-1bpw wxa16：MoE 86.5ms ≈ 全 step（128 experts top-8 无 shared）；
  triton 64.6、sort/scatter 6.0、router 2.9
- p99/mean 抖动 <15% → 负载不均影响有限；瓶颈是 dequant-GEMM kernel 本身

## graph decode

| ckpt | eager | graph replay | 加速 | tok/s | token_match |
|---|---|---|---|---|---|
| dsv2 wxa8（1030，128×64） | 48.6 ms/step | 17.4 ms/step | 2.78× | 57.2 | true |
| qwen3-1bpw wxa8（1031，128×128） | 79.1 ms/step | 28.4 ms/step | 2.79× | **35.22** | **false, DIFF@42** |
| qwen3-1bpw wxa8（1031，128×256） | 77.5 ms/step | 28.4 ms/step | 2.73× | 35.23 | false, DIFF@42 |

### DIFF@42 定性（benign tie-flip，证据链）

- `eager_top2_gap = 0.0`（精确为零，真平局）→ argmax 本身任意
- `graph_top2_gap = 0.1875 = logits_max_abs`（1 个 bf16 ulp，qwen3 logits
  幅度下 ulp ≈ 0.125–0.25）恰好打破平局
- `graph_logits_has_nan = False`；`prefill_parity = 0.0` 且 prefill token
  全等（graph 路径 prefill 段与 eager 逐位一致）；B2 复跑同一步确定性复现
- teacher-forced logits 对拍 max_abs = 4.9（dsv2 为 1.4）：48 层 × 256 步上
  静态派发 vs host 编排两种 kernel 排序的 bf16 累积噪声，软判据、无 NaN

## graph replay 逐 kernel profile（1101）

`profile_graph_decode.py`（torch profiler，graph replay 内 32 步平均），
qwen3-1bpw / dsv2-1bpw，wxa8：

| 桶 | qwen3-30b-a3b | deepseek-v2-lite |
|---|---|---|
| wall ms/step | 27.4（36.5 tok/s） | 16.0（62.6 tok/s） |
| kernel 合计 | 24.5 | 14.1 |
| elementwise 胶水 | **12.08（49%，8521 次/步）** | 6.47（46%，4729 次/步） |
| other（triton MoE 计算等） | 7.62（1301 次） | 3.97 |
| gemm | 3.69 | 3.18 |
| attention | 0.89 | 0.38 |
| memcpy | 0.19 | 0.09 |
| in-graph gap | 2.9（11%） | 1.9（12%） |

- qwen3 每步 ~10,403 kernel（每层 ~217）：graph 只消除 launch 间隙，胶水
  kernel 本体仍在跑 → **调优主线 = kernel 数量**（bit-pool 合 launch、
  gu/dn 融合），次线 triton 768 维配置（有效 162 GFLOPS vs dsv2 调过的
  241）
- 胶水占比两模型同为 46–49%：bit-partition 实现通病，合并优化五模型通吃
- b1 decode 每 token 仅读 top-8 激活专家（~3.3B 参 × ~1.7bit ≈ 0.7GB，
  5090 带宽下 ≪18ms）——无 memory 墙，纯 kernel 工程问题

## Qwen3.5 fork 55 tok/s 口径对账（1104，关闭）

fork 的 55.2 tok/s 出处：`DartMoQ-Qwen3.5/results/graph_decode_
Qwen3.5-35B-A3B_wxa8_72359b9+_261025_174656.json`（同款 5090、wxa8、
prompt 128 × 256 步、graph replay 18.0 ms/step）。与本项目 qwen3-30b-a3b
不可直接对标，三条口径差：

| 口径 | fork（55.2 来源） | 本项目 qwen3 |
|---|---|---|
| 模型 | Qwen3.5-35B-A3B：40 层（30 GDN 线性 + 10 full）、moe_inter 512 | qwen3-30b-a3b：48 层全 GQA、moe_inter 768 |
| MoE FLOPs/token | ~1.34 G | ~2.41 G（**1.8×**） |
| 测法 | graph replay | bench 表为 eager；graph 为 35.2 |

- eager 两边几乎一致（fork 76.9 vs 本项目 77.5–79.1 ms/step）：~10⁴
  kernel/步的 launch 墙掩盖模型轻重差异，port 本身无退化
- graph 内部 18.0 vs 28.4：主因模型计算量 1.8×（按 FLOPs 折 18.0×1.8≈
  32.4，本项目 28.4 单位 FLOPs 不亏），次因胶水未合并（12.08ms/49%）
- **结论**：55 不作为 qwen3-30b 目标口径，同栈标尺 dsv2 graph 57.2–62.6；
  qwen3 路径已量化：胶水 12.08→~4ms + triton 768 维调优 → 18–20ms/step
  ≈ 50–55 tok/s（P0 kernel 数量路线）

## 后边优化 TODO

1. **P0 kernel 数量优化**（2026-11-04 开工）：bit-pool 合 launch + gu/dn
   融合，目标 qwen3 胶水 12.08→~4ms（graph 35.2→~50 tok/s），dsv2 同吃
   （6.47ms）；按 cuda/triton 流程先写主流程对齐测试 + GPU 填充度/SM 分析
2. （次线）triton 768 维 shape 配置调优：162→241 GFLOPS 追平 dsv2 已调
   配置，估 +~15 tok/s；`test/test_p53_tune.py`、`test_wxa8_tune.py` 就绪
3. **对标基线格**：bf16 原模型（qwen3 57GiB 双卡可跑、单卡放不下——显存
   卖点句；prefill 2048×4 有 OOM 风险）；NF4 需 bitsandbytes 新版
   （sm_120）。`bench_throughput.py --baseline` 分支（未写）
4. **graph 数字扩展**：2bpw 档 + 其余模型（`GRAPH=1 bash run.bench.sh` 或
   run.graph.sh CKPTS 加行）
5. （可选）qwen3 teacher-forced parity 4.9 的 kernel 排序对齐观察项；
   B2 tie-flip 判据已修（1101），重跑 run.graph.sh 可出带新判定的存档


## 其它事实备查

- bench decode 为 **eager host 编排路径**（未设 DARTMOQ_STATIC_DISPATCH，
  bench_throughput.py:635 有 env 探针可证）；graph_decode.py 自设
  `DARTMOQ_STATIC_DISPATCH=1`（graph_decode.py:263），仅 wxa8
- qwen3-30b bf16 单卡不可用（57GiB>32GiB）；ours 1bpw 峰值 7.89GiB 单卡可跑
- run.graph.sh mask 默认 native，失败自动换 buffer 补跑（本次 native 直接过）
