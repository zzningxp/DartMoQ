# Roadmap: TurboQuant WxA16/WxA8 真量化加速移植（Qwen3.5 fork → DartMoQ）

> 调研日期：2026-09-17
> 源仓库：`../DartMoQ-Qwen3.5/`（从本仓库 `fae0532` 2026-07-02 分叉，之后做了 100+ 提交完成 turboquant triton WxA16/WxA8 加速与真量化 checkpoint 保存）

## 目标

为 DartMoQ 支持的五个 MoE 模型（olmoe、dsv1、dsv2、moon、qwen3-30b-a3b）移植：

1. turboquant triton WxA16（混合 bit packed 权重 + fp16 激活 + fp16 tensor core）与 WxA8（int8 激活 + int8 tensor core）加速推理
2. 真实量化参数保存/加载（packed safetensors + meta.json）
3. 直接 a8/a16 推理与速度测试

## 已确认的决策（2026-09-17 与用户讨论确认）

- qwen3 范围：只做 Qwen3-30B-A3B（MoE）；dense（0.6B/4B/8B）保持现状仅 eval
- 管线接入：新 packed 真量化路径与现有 fake-quant/gptq 路径**并存**，`--save-quantized` 自动启用真量化
- 节奏：WxA16 与 WxA8 一起移植（共用 checkpoint 格式与模块继承结构）
- 不移植 qwen3.5 特有 attention（delta_rule.py、norm_kernels.py、attn_profile.py）

## 调研关键结论

- 两边 `turboquant_utils/` 只有 3 个文件有差异：triton_kernels.py（345→1350 行）、quantize.py（274→817 行）、rotation.py（158→306 行），其余一致
- 内核完全通用：只依赖张量形状和 packed 布局（indices_packed_gf uint8 + fp16 码本 + norms + seed），不感知模型结构
- WxA8 与 WxA16 共用同一份 checkpoint：A8 是运行时类切换（继承 A16 类只覆盖 gate_up/down matmul），加载期 fp16 码本转 int8，零拷贝
- checkpoint 格式：`model.safetensors`（`bit_weights.{bit}.{gate_up|down}_{indices_packed|codebook|norms}`、attn packed、fp16 未量化参数、qmeta seed 冗余）+ `meta.json`（per-layer bit/group_size/seed/rotation/codebook_type、MoE expert 结构）
- DartMoQ 微专家结构（DartMoQHybridWrapper）= intermediate 维行切片，可自然映射到 bit 分桶 grouped-gemm 布局，DP/rank-mode 逻辑无需改动
- 五模型 mlp 结构：dsv1/dsv2/moon/qwen3moe 有 shared expert，olmoe 无；替换 mlp 时复用原 gate + shared expert 子模块

## 阶段计划

### 阶段 1：内核层（turboquant_utils）
- [x] 替换 triton_kernels.py / quantize.py / rotation.py 为 Qwen3.5 版本
- [x] 新增 triton_kernels_a8.py、kernel_autotune.py、cuda_profiler.py
- [x] 移植 test_triton_mixed_precision.py、test_triton_mp_moe_e2e_bench.py（e2e 依赖阶段 2 的 quantization 包，阶段 2 后跑）
- [x] 移植内核级测试 test_wxa8_kernel.py / test_wxa8_fused_quant.py / test_wxa8_cfg_compare.py / test_wxa8_tune.py / test_group_first_layout.py / test_p53_tune.py / wxa8_common.py
- [x] 核对 module.py TurboQuantLinear 对旧内核接口的调用兼容性（triton_fused_matmul/dual 均保留；viz/ 脚本用到的 turboquant_quantize/generate_rotation_matrix/hadamard_rotate 均保留）
- 测试结果（2026-09-17，dart312 环境）：A16 内核对拍通过（vs 反量化+GEMM 参考）；A8 内核对拍通过（kernel 级 1.52x：gate_up 1.59x / down 1.40x，含 SM 填充度分析）；rotate+quantize 融合对拍通过（hoist 场景 18.77x）；group-first 布局测试通过

### 阶段 2：量化模块层（新建 quantization/ 包）
- [x] quantization/wxa16/：linear.py、bit_partitioned_moe.py（WxA16Weights + BitPartitionedGroupMoE）、量化入口、memory_stats.py
- [x] quantization/wxa8/：linear.py、bit_partitioned_moe.py（继承 A16，覆盖 matmul 两处）
- [x] 适配五模型：新增 quantization/common/adapters.py 行为约定层（router_mode: deepseek_tuple3/tuple2/qwen3_moe/softmax_topk；shared_mode: gated/sum/none），按 model_type 一行映射，写入 meta.json 自描述；deepseek 家族 gate 喂 3D 输入；根目录 bit_partitioned_moe.py（fp16 中间版）补 shared_experts/top_k 透传
- [x] 删除 _build_hoisted_rotations 中结果被覆盖的死循环（原版合并残留，实际生效的是第二个循环，行为不变）

### 阶段 3：管线接线 + 保存加载
- [x] dartmoq_layer_reconstruct.py：记录 per-expert bit_to_indices；构建 layer_metadata（num_experts/hidden/inter/bit_list/top_k/router_mode/shared_mode）；返回 (moe, layer_metadata)；hybrid 分支补 shared_expert/shared_expert_gate 复制（原版只复制 shared_experts，qwen3 系门控共享专家会丢——附带修复）
- [x] dartmoq_hybridmoe.py：DartMoQHybridWrapper 加 bit_to_indices 参数
- [x] dartmoq_utils.py：quant_layer_mix_precision 加 update_weight 参数（0-bit/turboquant 分支）；新增 collect_all_linears_recursive + quant_layer_mix_precision_wxa16（attention 递归收集含 MLA 嵌套、shared_expert/shared_experts 双结构、非 MoE 层标准 MLP 按 share 位量化）
- [x] dartmoq_sequential.py：construct_moe 接线（use_wxa16 = --wxa16 或 --save-quantized；quant 分派；from_build_block 重组 + 旧结构清理；packed 路径要求 hybrid 模式）
- [x] 新建 dartmoq_quant_io.py（save/load/restore/convert_to_wxa8/collect_metadata + qmeta 种子核对 + auto_map 自定义代码拷贝）
- [x] run_dartmoq.py 加 `--wxa16`/`--save-quantized`；eval_dartmoq.py 加 `--load-quantized` + `--inference-quant-mode {wxa16,wxa8}`（位置参数自动识别量化目录）
- [x] 验证 dsv1/dsv2/moon 自定义 modeling 的 device_map="meta" 加载兼容性（test/test_meta_skeleton.py，五模型全过）
- [x] bf16 激活全链路验证（五个模型 checkpoint 为 bf16，qwen3.5 为 fp16——已单独验证 dtype 兼容）

### 阶段 4：测速与验证
- [x] 内核对拍测试（test_wxa8_kernel 1.52x / test_triton_mixed_precision / test_wxa8_fused_quant 18.77x / test_group_first_layout / test_wxa8_linear 安全阀 / test_wxa8_moe A8 vs A16 relerr~0.010-0.014）
- [x] 端到端 bench（test_triton_mp_moe_e2e_bench，小形状跑通，数值误差 0.14%；真实形状需按五模型配置 --bits/--hidden-size 等参数）
- [x] round-trip 测试（test/test_quant_io.py：gated/sum 双 shared 模式 + tuple3 gate + A8 转换，全过；A8 MoE relerr 0.014）
- [x] 全模型量化+保存：五个模型 2bpw checkpoint 全部生成并 eval 通过（2026-09-20，run.q.sh）：
  olmoe wiki 12.6387/c4 17.9668、dsv1 7.409/11.7952、dsv2 7.0075/11.2046、
  moon 8.6242/19.5983、qwen3-30b 9.2282/13.7167（2.3G/5.5G/5.3G/5.8G/9.4G）
- [x] wxa16/wxa8 推理测速 vs fp16 baseline（2026-09-21，run.e.sh 全量完成，三模式同 sequential + 32 批量口径）：
  | 模型 | fp16 | wxa16 | wxa8 | wxa16 vs fp16 | wxa8 vs fp16 |
  |---|---|---|---|---|---|
  | olmoe | 97.3s | 83.0s | 75.3s | -14.7% | -22.7% |
  | dsv1 | 194.2s | 151.3s | 129.7s | -22.1% | -33.2% |
  | dsv2 | 254.7s | 200.6s | 168.2s | -21.3% | -34.0% |
  | moon | 182.2s | 194.7s | 160.9s | +6.8% | -11.7% |
  | qwen3-30b | 377.1s | 261.9s | 230.7s | -30.5% | -38.8% |
  ppl 全部与量化时一致（如 moon 8.6260 vs 8.6242、qwen3 9.2268 vs 9.2282）。
  moon wxa16 的 c4（120s vs fp16 90s）是唯一未过线的点，c4 路由分散场景待优化。
- [ ] autotune 驱动（test/test_p53_tune.py / test_wxa8_tune.py 已就绪，按五模型真实形状跑）
- [ ] moon wxa16 c4 场景优化（活跃专家/bit 多、内核启动密集；CudaStageProfiler 定位 + 配置重调）
- [ ] 遗留：_build_hoisted_rotations 上游原版有结果被覆盖的死循环已删；packed 字典缓存跨设备移动的陈旧引用与上游同构（load 路径 cache=None 惰性重建，安全）

## 修复记录（2026-09-19/20）

1. **convert_to_group_first 单 group norms 形状 bug**（上游同源 bug，qwen3.5 模型形状下从未触发）：
   单 group 时 packed 量化的 norms 是 1-D (N,)，`norms.t()` 不变形导致 down 1-group
   回退路径 `norms_slice[0]` 取到标量 → 输出数值爆炸（inf/NaN）。已修复（unsqueeze 后转置），
   直接验证 diff 0.047（fp16 舍入级）。触发场景：某 bit 分桶总神经元数 == group_size
   （如稀有位宽 bit=1 或小模型的 bit 桶）。A8 down 同根因一并修复。
2. **drop_runtime_caches 显存滞留修复**：standby 流程 layer.to('cpu') 只搬注册 buffer，
   `_packed` 字典与 gf 布局缓存持有旧 GPU 张量引用 → 每层滞留约一份 packed 权重显存。
   WxA16BitPartitionedGroupMoE / WxA8Linear 新增 drop_runtime_caches()，
   在 dartmoq_sequential（量化侧）与 eval_dartmoq cmoe_ppl_eval_sequential（评估侧）
   的 layer.to('cpu') 后调用；forward 按当前 buffer 设备惰性重建（已验证 drop+rebuild
   数值与首次一致）。
3. 脚本：run.q.sh（五模型 2bpw 量化保存，quant_ckpt/<name>-2bpw 独立子目录）、
   run.e.sh（五模型 × fp16/wxa16/wxa8 三模式速度对比）。
4. bf16 激活全链路验证补充：五个模型 checkpoint 为 bf16（qwen3.5 为 fp16），
   已单独验证 bf16 输入下 forward 数值正常（此前仅 fp32 覆盖）。
5. **tool_utils.force_release_inactive_splits 设备切换不恢复**（既有 bug，triton 才暴露）：
   set_device(1) 后不恢复，standby 双卡初始化后进程当前设备停在 cuda:1，
   triton kernel 用 cuda:0 张量 launch → cuPointerGetAttribute INVALID_VALUE →
   报 "Pointer argument ... (cpu tensor?)"。已修（函数末尾恢复原设备）。
6. **convert_to_group_first 单 group norms**、**0-bit/3-bit 位宽**、**128 对齐自动 slices**
   修复见上（2026-09-19/20 记录）。
7. qwen3-30b 首次运行层 37 "CUDA error: unknown error" 为驱动挂死前兆（重启后
   同参数全量通过）；已加每层结束 torch.cuda.synchronize() 便于异步 kernel 错误
   当场定位。
8. cmoe_ppl_eval 的 standby 参数名修复：同时认 standby_layer_cpu / standby_cpu，
   fp16 基线 CPU standby 时强制 sequential eval。

## 风险点

1. dsv1/dsv2/moon 自定义 modeling 文件的 device_map="meta" 加载兼容性（需小测试验证）
2. 旧 TurboQuantLinear 对旧内核接口的依赖
3. hybrid 模式微专家切片 → bit 分桶的映射正确性（0-bit 专家不进桶）
4. 每层处理完及时释放 GPU/CPU 内存；首次运行 Triton JIT 编译风暴（warmup 预热）

## 手动测试（全模型主流程，2026-09-19）

建议从 olmoe（最小，1B 激活）开始验证，再逐个模型跑。主流程环境建议 cmoe311
（transformers 4.57.5，checkpoint 为 ModuleList 专家格式；dart312 的 transformers
5.x 已改为 3D 专家格式，与现有 checkpoint 不兼容）。

```bash
# 1. 量化 + 保存真实量化参数（--save-quantized 自动启用 WxA16 真量化路径）
python run_dartmoq.py models/OLMoE-1B-7B-0924-Instruct wikitext2 \
    --nsamples 32 --slices 4 --quant-scheme global-a8s4m2bpw \
    --rank-mode turboquant_innerproduct --quantmode turboquant \
    --save-quantized ./quant_ckpt_olmoe

# 2. 加载量化 checkpoint 直接推理（wxa16：fp16 激活）
python eval_dartmoq.py --load-quantized ./quant_ckpt_olmoe --inference-quant-mode wxa16

# 3. wxa8（int8 激活，同一份 checkpoint）
python eval_dartmoq.py --load-quantized ./quant_ckpt_olmoe --inference-quant-mode wxa8

# 4. fp16 基线对比
python eval_dartmoq.py models/OLMoE-1B-7B-0924-Instruct

# 5. 大模型用 sequential + standby 模式
python eval_dartmoq.py --load-quantized ./quant_ckpt_dsv2 --sequential-eval --standby-cpu

# 6. 端到端 bench（按目标模型形状）
python turboquant_utils/test_triton_mp_moe_e2e_bench.py \
    --num-experts 64 --top-k 6 --hidden-size 2048 --intermediate-size 1408 \
    --bits "2:704,4:704" --batch-size 1 --seq-len 2048

# 7. 离线 autotune（按真实 eval 形状网格搜索，结果需回填配置表）
python test/test_p53_tune.py          # A16
python test/test_wxa8_tune.py         # A8
```

首次运行 wxa8 会触发 Triton JIT 编译（逐 expert 形状边跑边编译），测速前先跑一次任意
wxa8 eval 捂热磁盘缓存。

## 测试约定

- 小测试自己跑：`conda run -n dart312 python test/xxx.py`；HF 数据集离线 `export HF_DATASETS_OFFLINE=1`
- 全模型主流程（run_dartmoq.py / eval_dartmoq.py）不自动跑，由本人手动测试，每次开发后提供手动命令
- git 提交均由本人手动操作
