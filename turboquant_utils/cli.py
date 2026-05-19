"""CLI for TurboQuant model quantization and evaluation.

Usage:
    # Single-pass 4-bit quantization
    turboquant quantize --model Qwen/Qwen3.5-0.8B-Base --output ./quantized --bit-width 4

    # Residual quantization (4+4 = 8 total bits)
    turboquant quantize --model Qwen/Qwen3.5-0.8B-Base --output ./quantized \
        --bit-width 4 --residual-bit-width 4

    # Evaluate PPL
    turboquant eval --model Qwen/Qwen3.5-0.8B-Base --quantized ./quantized

    # Evaluate PPL + KL divergence vs. reference model
    turboquant eval --model Qwen/Qwen3.5-0.8B-Base --quantized ./quantized --kld

    # Run inference
    turboquant generate --model Qwen/Qwen3.5-0.8B-Base --quantized ./quantized \
        --prompt "The capital of France is"
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

logger = logging.getLogger("turboquant_model")


def _auto_device() -> str:
    """Auto-detect the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _disable_fused_kernels(model):
    """Disable GPU-only fused kernels (cuTile, Triton). Preserves Metal on macOS."""
    from turboquant_model.module import TurboQuantLinear
    for m in model.modules():
        if isinstance(m, TurboQuantLinear):
            m.use_cutile = False
            m.use_triton = False


def cmd_quantize(args: argparse.Namespace):
    """Quantize a model and save to disk."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from turboquant_model.model import TurboQuantConfig, quantize_model, quantize_model_advanced, save_quantized

    device = args.device or _auto_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    logger.info(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True
    ).to(device).eval()

    config = TurboQuantConfig(
        bit_width=args.bit_width,
        group_size=args.group_size,
        seed=args.seed,
        skip_embeddings=args.skip_embeddings,
        skip_lm_head=args.skip_lm_head,
        residual_bit_width=args.residual_bit_width,
        residual_seed=args.residual_seed,
        rotation=args.rotation,
        rotation_strategy=args.rotation_strategy,
        norm_codec=getattr(args, 'norm_codec', 'fp32') or 'fp32',
        entropy_coding=getattr(args, 'entropy_coding', False),
        cpu_offload_pass2=getattr(args, 'cpu_offload_pass2', False),
    )

    logger.info(f"Quantizing: {config.bit_width}-bit"
                + (f"+{config.residual_bit_width}-bit residual" if config.residual_bit_width else "")
                + (f" norm_codec={config.norm_codec}" if config.norm_codec != "fp32" else "")
                + (" +entropy_coding" if config.entropy_coding else ""))
    if config.norm_codec != "fp32":
        model = quantize_model_advanced(model, config)
    else:
        model = quantize_model(model, config)

    if device == "cpu":
        _disable_fused_kernels(model)
    elif device == "mps":
        _disable_fused_kernels(model)  # keep Metal

    # Norm calibration (requires the fp model as reference)
    if getattr(args, 'calibrate', False):
        from turboquant_model.norm_calibration import calibrate_norms_blockwise, CalibrationConfig

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        logger.info("Loading reference model for norm calibration...")
        fp_model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, trust_remote_code=True
        ).to(device).eval()

        cal_config = CalibrationConfig(
            n_samples=getattr(args, 'calibrate_samples', 4),
            lam=getattr(args, 'calibrate_lambda', 1.0),
            lr=getattr(args, 'calibrate_lr', 1e-3),
            n_iters=getattr(args, 'calibrate_iters', 50),
        )
        logger.info(f"Running blockwise norm calibration ({cal_config.n_samples} samples, "
                    f"{cal_config.n_iters} iters, lambda={cal_config.lam})")
        calibrate_norms_blockwise(model, fp_model, tokenizer, device=device, config=cal_config)
        del fp_model
        if device == "cuda":
            torch.cuda.empty_cache()

    logger.info(f"Saving to: {args.output}")
    save_quantized(model, config, args.output)

    # Quick sanity check
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    input_ids = tokenizer.encode("Hello world", return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(input_ids).logits
    logger.info(f"Sanity check passed: logits shape {logits.shape}")


def cmd_eval(args: argparse.Namespace):
    """Evaluate PPL (and optionally KL divergence) on WikiText-103."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    device = args.device or _auto_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if args.quantized:
        from turboquant_model.model import load_quantized
        offload = getattr(args, 'cpu_offload_pass2', None) or None
        logger.info(f"Loading quantized model from: {args.quantized}")
        model = load_quantized(args.model, args.quantized, device=device, cpu_offload_pass2=offload)
    else:
        from turboquant_model.model import TurboQuantConfig, quantize_model
        logger.info(f"Loading and quantizing model: {args.model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, trust_remote_code=True
        ).to(device).eval()
        config = TurboQuantConfig(
            bit_width=args.bit_width,
            group_size=args.group_size,
            seed=args.seed,
            residual_bit_width=args.residual_bit_width,
            residual_seed=args.residual_seed,
            rotation=args.rotation,
            rotation_strategy=getattr(args, "rotation_strategy", "different"),
            cpu_offload_pass2=getattr(args, 'cpu_offload_pass2', False),
        )
        model = quantize_model(model, config)

    if device in ("cpu", "mps"):
        _disable_fused_kernels(model)

    # Load reference model for KLD computation
    ref_model = None
    if args.kld:
        logger.info(f"Loading reference model for KLD: {args.model}")
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, trust_remote_code=True
        ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load WikiText-103 validation
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    text = "\n\n".join(ds["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]

    seq_len = args.seq_length
    n_chunks = min(args.n_chunks, (len(input_ids) - 1) // seq_len)

    total_loss = 0.0
    total_tokens = 0
    total_kld = 0.0

    model.eval()
    with torch.no_grad():
        for i in range(n_chunks):
            start = i * seq_len
            chunk = input_ids[start : start + seq_len + 1].unsqueeze(0).to(device)
            outputs = model(chunk[:, :-1])
            logits = outputs.logits
            targets = chunk[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += targets.numel()

            if ref_model is not None:
                ref_logits = ref_model(chunk[:, :-1]).logits
                log_p = torch.nn.functional.log_softmax(logits, dim=-1)
                log_q = torch.nn.functional.log_softmax(ref_logits, dim=-1)
                kld = torch.nn.functional.kl_div(
                    log_p.reshape(-1, logits.shape[-1]),
                    log_q.reshape(-1, logits.shape[-1]),
                    log_target=True,
                    reduction="sum",
                )
                total_kld += kld.item()

    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    logger.info(f"PPL ({n_chunks} chunks, seq_len={seq_len}): {ppl:.4f}")
    print(f"PPL: {ppl:.4f}")

    if ref_model is not None:
        avg_kld = total_kld / total_tokens
        logger.info(f"KLD ({n_chunks} chunks, seq_len={seq_len}): {avg_kld:.6f}")
        print(f"KLD: {avg_kld:.6f}")


def cmd_generate(args: argparse.Namespace):
    """Generate text with a quantized model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or _auto_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if args.quantized:
        from turboquant_model.model import load_quantized
        offload = getattr(args, 'cpu_offload_pass2', None) or None
        model = load_quantized(args.model, args.quantized, device=device, cpu_offload_pass2=offload)
    else:
        from turboquant_model.model import TurboQuantConfig, quantize_model
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, trust_remote_code=True
        ).to(device).eval()
        config = TurboQuantConfig(
            bit_width=args.bit_width,
            residual_bit_width=args.residual_bit_width,
            cpu_offload_pass2=getattr(args, 'cpu_offload_pass2', False),
        )
        model = quantize_model(model, config)

    if device in ("cpu", "mps"):
        _disable_fused_kernels(model)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1e-6),
        )
    elapsed = time.time() - t0

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    n_tokens = output.shape[1] - input_ids.shape[1]

    print(text)
    logger.info(f"Generated {n_tokens} tokens in {elapsed:.1f}s ({n_tokens/elapsed:.1f} tok/s)")


def cmd_calibrate(args: argparse.Namespace):
    """Calibrate norms of a saved quantized model (post-hoc)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turboquant_model.model import load_quantized, save_quantized, TurboQuantConfig
    from turboquant_model.norm_calibration import calibrate_norms_blockwise, CalibrationConfig

    device = args.device or _auto_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    logger.info(f"Loading quantized model from: {args.quantized}")
    model = load_quantized(args.model, args.quantized, device=device)

    if device in ("cpu", "mps"):
        _disable_fused_kernels(model)

    logger.info(f"Loading reference model: {args.model}")
    fp_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    cal_config = CalibrationConfig(
        n_samples=args.n_samples,
        seq_length=args.seq_length,
        lam=args.lam,
        lr=args.lr,
        n_iters=args.n_iters,
        batch_size=args.batch_size,
    )
    logger.info(f"Calibrating norms blockwise ({cal_config.n_samples} samples, "
                f"{cal_config.n_iters} iters, lambda={cal_config.lam})")
    results = calibrate_norms_blockwise(model, fp_model, tokenizer, device=device, config=cal_config)

    # Print summary
    avg_mse_before = sum(r["before_mse"] for r in results) / len(results)
    avg_mse_after = sum(r["after_mse"] for r in results) / len(results)
    avg_cos_before = sum(r["before_cos"] for r in results) / len(results)
    avg_cos_after = sum(r["after_cos"] for r in results) / len(results)
    print(f"\nCalibrated {len(results)} layers:")
    print(f"  Avg MSE:    {avg_mse_before:.6f} -> {avg_mse_after:.6f}")
    print(f"  Avg cos_sim: {avg_cos_before:.6f} -> {avg_cos_after:.6f}")

    # Save if output specified
    output = getattr(args, 'output', None)
    if output:
        config = TurboQuantConfig.load(
            Path(args.quantized) / "turboquant_config.json"
        )
        save_quantized(model, config, output)
        logger.info(f"Saved calibrated model to: {output}")
    else:
        # Overwrite in-place
        config = TurboQuantConfig.load(
            Path(args.quantized) / "turboquant_config.json"
        )
        save_quantized(model, config, args.quantized)
        logger.info(f"Saved calibrated model to: {args.quantized} (overwritten)")


def cmd_benchmark(args: argparse.Namespace):
    """Benchmark memory and latency."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from turboquant_model.model import TurboQuantConfig, quantize_model

    device = args.device or _auto_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    config = TurboQuantConfig(
        bit_width=args.bit_width,
        residual_bit_width=args.residual_bit_width,
        cpu_offload_pass2=getattr(args, 'cpu_offload_pass2', False),
    )
    model = quantize_model(model, config)

    if device in ("cpu", "mps"):
        _disable_fused_kernels(model)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    input_ids = tokenizer.encode("The quick brown fox", return_tensors="pt").to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model(input_ids)

    # Benchmark
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(args.n_iters):
            model(input_ids)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    if device == "cuda":
        mem_base = torch.cuda.memory_allocated() / 1024**2
        mem_peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Memory: {mem_base:.0f} MB base, {mem_peak:.0f} MB peak")
    else:
        try:
            import psutil, os
            rss = psutil.Process(os.getpid()).memory_info().rss / 1024**2
            print(f"Memory (RSS): {rss:.0f} MB")
        except ImportError:
            print("Memory: psutil not available")
    print(f"Latency: {elapsed / args.n_iters * 1000:.1f} ms/forward")


def main():
    parser = argparse.ArgumentParser(
        prog="turboquant",
        description="TurboQuant: Near-optimal weight quantization with on-the-fly dequantization",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- quantize ---
    p_quant = subparsers.add_parser("quantize", help="Quantize a model and save")
    p_quant.add_argument("--model", required=True, help="HF model name or path")
    p_quant.add_argument("--output", required=True, help="Output directory")
    p_quant.add_argument("--device", default=None, help="Device: cuda, mps, or cpu (auto-detected)")
    p_quant.add_argument("--bit-width", type=int, default=4)
    p_quant.add_argument("--group-size", type=int, default=None)
    p_quant.add_argument("--seed", type=int, default=42)
    p_quant.add_argument("--skip-embeddings", action="store_true")
    p_quant.add_argument("--skip-lm-head", action="store_true")
    p_quant.add_argument("--residual-bit-width", type=int, default=None, help="Bits for residual pass")
    p_quant.add_argument("--residual-seed", type=int, default=1042)
    p_quant.add_argument("--rotation", choices=["qr", "hadamard"], default="qr",
                         help="Rotation method: qr (Haar random) or hadamard (fast Walsh-Hadamard)")
    p_quant.add_argument("--rotation-strategy", choices=["different", "shared", "alternating"],
                         default="different",
                         help="Rotation strategy for residual: different (default, best quality), "
                              "shared (enables merge_and_requantize), alternating (for multi-pass)")
    p_quant.add_argument("--norm-codec", choices=["fp32", "fp16", "factored_int8", "factored_int4"],
                         default="fp32", help="Norm compression method")
    p_quant.add_argument("--entropy-coding", action="store_true",
                         help="Enable rANS entropy coding of quantized indices")
    p_quant.add_argument("--cpu-offload-pass2", action="store_true",
                         help="Offload residual pass2 weights to CPU with pipelined H2D")
    p_quant.add_argument("--calibrate", action="store_true",
                         help="Run norm calibration after quantization (requires fp model in memory)")
    p_quant.add_argument("--calibrate-samples", type=int, default=4,
                         help="Number of calibration sequences")
    p_quant.add_argument("--calibrate-lambda", type=float, default=1.0,
                         help="Lambda weight for angular+KLD loss in calibration")
    p_quant.add_argument("--calibrate-lr", type=float, default=1e-3,
                         help="Learning rate for norm calibration")
    p_quant.add_argument("--calibrate-iters", type=int, default=50,
                         help="Number of optimisation iterations per block")

    # --- eval ---
    p_eval = subparsers.add_parser("eval", help="Evaluate PPL on WikiText-103")
    p_eval.add_argument("--model", required=True, help="HF model name or path")
    p_eval.add_argument("--device", default=None, help="Device: cuda, mps, or cpu (auto-detected)")
    p_eval.add_argument("--quantized", default=None, help="Quantized model dir (skip live quant)")
    p_eval.add_argument("--bit-width", type=int, default=4)
    p_eval.add_argument("--group-size", type=int, default=None)
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.add_argument("--residual-bit-width", type=int, default=None)
    p_eval.add_argument("--residual-seed", type=int, default=1042)
    p_eval.add_argument("--seq-length", type=int, default=512)
    p_eval.add_argument("--n-chunks", type=int, default=50)
    p_eval.add_argument("--kld", action="store_true", help="Compute KL divergence vs. reference model")
    p_eval.add_argument("--rotation", choices=["qr", "hadamard"], default="qr",
                        help="Rotation method: qr or hadamard")
    p_eval.add_argument("--rotation-strategy", choices=["different", "shared", "alternating"],
                        default="different",
                        help="Rotation strategy for residual passes")
    p_eval.add_argument("--cpu-offload-pass2", action="store_true",
                        help="Offload residual pass2 weights to CPU with pipelined H2D")

    # --- generate ---
    p_gen = subparsers.add_parser("generate", help="Generate text")
    p_gen.add_argument("--model", required=True, help="HF model name or path")
    p_gen.add_argument("--device", default=None, help="Device: cuda, mps, or cpu (auto-detected)")
    p_gen.add_argument("--quantized", default=None, help="Quantized model dir")
    p_gen.add_argument("--bit-width", type=int, default=4)
    p_gen.add_argument("--residual-bit-width", type=int, default=None)
    p_gen.add_argument("--prompt", required=True)
    p_gen.add_argument("--max-tokens", type=int, default=64)
    p_gen.add_argument("--temperature", type=float, default=0.0)
    p_gen.add_argument("--cpu-offload-pass2", action="store_true",
                       help="Offload residual pass2 weights to CPU with pipelined H2D")

    # --- benchmark ---
    p_bench = subparsers.add_parser("benchmark", help="Benchmark memory and latency")
    p_bench.add_argument("--model", required=True, help="HF model name or path")
    p_bench.add_argument("--device", default=None, help="Device: cuda, mps, or cpu (auto-detected)")
    p_bench.add_argument("--bit-width", type=int, default=4)
    p_bench.add_argument("--residual-bit-width", type=int, default=None)
    p_bench.add_argument("--n-iters", type=int, default=10)
    p_bench.add_argument("--cpu-offload-pass2", action="store_true",
                         help="Offload residual pass2 weights to CPU with pipelined H2D")

    # --- calibrate ---
    p_cal = subparsers.add_parser("calibrate", help="Calibrate norms of a saved quantized model")
    p_cal.add_argument("--model", required=True, help="HF model name or path (reference)")
    p_cal.add_argument("--quantized", required=True, help="Quantized model directory")
    p_cal.add_argument("--output", default=None, help="Output dir (default: overwrite quantized)")
    p_cal.add_argument("--device", default=None, help="Device: cuda, mps, or cpu (auto-detected)")
    p_cal.add_argument("--n-samples", type=int, default=4, help="Number of calibration sequences")
    p_cal.add_argument("--seq-length", type=int, default=2048, help="Sequence length")
    p_cal.add_argument("--lam", type=float, default=1.0, help="Lambda for angular+KLD loss")
    p_cal.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p_cal.add_argument("--n-iters", type=int, default=50, help="Iterations per block")
    p_cal.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    commands = {
        "quantize": cmd_quantize,
        "eval": cmd_eval,
        "generate": cmd_generate,
        "benchmark": cmd_benchmark,
        "calibrate": cmd_calibrate,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
