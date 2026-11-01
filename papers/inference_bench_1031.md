# 推理速度/显存 rebuttal 数据（1031 bench + graph decode）

- 目的：补齐评审要求的口径 —— checkpoint GB、含 metadata/mask 的 effective bpw、
  峰值推理显存、prefill/decode 延迟、tokens/s、不同 batch/seq 吞吐；并回应
  "MoE 只访问 routed experts、混合精度解码、不规则调度、负载不均可能抵消带宽节省"
- 环境：RTX 5090 ×2（32GiB each），torch 2.9.0+cu128，triton 3.5.0，
  transformers 5.20.1，commit `5d6b6e2+`
- 日志/结果：
  - bench：`logs/1031.e4b2_bench.log` + `results/throughput_*.json`（8 份），
    marker `logs/bench_done/`，入口 `run.bench.sh`（`bench_throughput.py`）
  - graph：`logs/1031.graph_qwen3.log` +
    `results/graph_decode_qwen3-30b-a3b_wxa8_5d6b6e2+_261031_233129.json`，
    入口 `run.graph.sh`（`graph_decode.py`）
- 状态（2026-10-31）：bench 8/20 格完成（dsv2×4、qwen3×4，一致性检查全 PASS）；
  qwen3-1bpw graph decode 完成（tie-flip 待优化判定，见下）

## bench 已完成格（8 格）

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

## 后边优化 TODO

1. ~~**B2 tie-flip 判据修正**~~（2026-11-01 已修）：绝对阈值 1e-2 → 参考
   top-1 幅度下 bf16 ulp 的 4 倍（2^(exp-8)，div 新增 eager_top1_exp 键，
   旧格式回落 1e-2 绝对判据保持兼容）；判定逻辑抽成
   graph_decode.py 的 classify_b2_divergence，测试
   test/test_graph_b2_verdict.py 四组 PASS（含 nextafter 实测 bf16 间距
   锚定 ulp 公式）。qwen3 1031 实测值在 exp∈{4,5,6} 下均判
   "tie-flip (likely benign)"；旧 JSON 不含 exp 键，重跑 run.graph.sh
   （删 marker logs/graph_done/qwen3-1bpw_wxa8）可出带新判定的存档
2. **bench 剩余 12 格续跑**（olmoe×4、dsv1×2、moon×2 及各 mode）：两个 bug
   已修——① adapters.py softmax_topk 分支支持 transformers 5.20
   OlmoeTopKRouter 3 元组 `(router_logits, router_scores, router_indices)`
  （scores 已 topk+renorm，直接采用；测试 `test/test_router_adapters.py`
   三项 PASS）；② run.bench.sh 加第三列 flags，dsv1/moon 带
   `--trust-remote-code`（DeepseekForCausalLM 无原生支持；moon 为
   deepseek_v3 有原生但量化时用 trust，先 trust 稳妥）。手动
   `bash run.bench.sh` 即续跑（marker 跳过已完成 8 格）
3. **对标基线格**：bf16 原模型（qwen3 57GiB，双卡 64GiB 可跑、单卡放不下
   ——本身是显存卖点句；prefill 2048×4 有 OOM 风险）；NF4（bnb）单卡 17GiB，
   但 dart312 未装 bitsandbytes（5090=sm_120 需新版）。需先给
   bench_throughput.py 加 `--baseline` 加载分支（尚未写）
4. **graph 数字扩展**：`GRAPH=1 bash run.bench.sh` 或往 run.graph.sh 的
   CKPTS 加行，补 dsv2-2bpw / qwen3-2bpw / olmob 等；qwen3-2bpw 预期同样
   2.7× 量级
5. （可选）qwen3 teacher-forced parity 4.9 的 kernel 排序对齐观察项

## 其它事实备查

- bench decode 为 **eager host 编排路径**（未设 DARTMOQ_STATIC_DISPATCH，
  bench_throughput.py:635 有 env 探针可证）；graph_decode.py 自设
  `DARTMOQ_STATIC_DISPATCH=1`（graph_decode.py:263），仅 wxa8
- qwen3-30b bf16 单卡不可用（57GiB>32GiB）；ours 1bpw 峰值 7.89GiB 单卡可跑
- run.graph.sh mask 默认 native，失败自动换 buffer 补跑（本次 native 直接过）
