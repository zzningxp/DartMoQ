"""DP assigned loss by sorted sub-expert order view.

Default parameters:
  model: qwen3-30b-a3b
  quantmode: turboquant
  rank_mode: turboquant_innerproduct
  layers: 7
  bits: 0 1 2 3 4
  slices_per_expert: 8
  sort_bit: lowest requested available bit, usually b0
  target_bpw: 2.0
  disable_0bit_compensation: true, for faster plotting
  out_dir: figs/dp_score_views

Meaning:
  y = sum(q_rates[assigned_bit]) * expert_activation_rate.
  x = DP sorted sub-expert index, colored by assigned bit.

Ready-to-run command:
  python -m viz.dp_score_tests.exp_5.plot_exp5_assigned_loss_by_order \
    --model qwen3-30b-a3b \
    --quantmode turboquant \
    --rank-mode turboquant_innerproduct \
    --layers 7 \
    --include-random-layer \
    --random-seed 123 \
    --bits 0 1 2 3 4 \
    --slices-per-expert 8 \
    --target-bpw 2.0 \
    --disable-0bit-compensation \
    --out-dir figs/dp_score_views

Strict 0bit-compensation variant, slower:
  python -m viz.dp_score_tests.exp_5.plot_exp5_assigned_loss_by_order \
    --model qwen3-30b-a3b \
    --quantmode turboquant \
    --rank-mode turboquant_innerproduct \
    --layers 7 \
    --bits 0 1 2 3 4 \
    --slices-per-expert 8 \
    --target-bpw 2.0 \
    --enable-0bit-compensation \
    --out-dir figs/dp_score_views
"""

from __future__ import annotations

import argparse
import os

from viz.micro_expert_rank_boxplot import (
    choose_random_layer,
    discover_layers,
    resolve_cache_dir,
    resolve_model_id,
    run_for_layer,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DP assigned loss along sorted sub-expert order.")
    parser.add_argument("--model", default="qwen3-30b-a3b")
    parser.add_argument("--quantmode", default="turboquant", choices=["turboquant", "gptq"])
    parser.add_argument("--rank-mode", "--rank_mode", dest="rank_mode", default="turboquant_innerproduct")
    parser.add_argument("--cache-root", default="auto")
    parser.add_argument("--activation-root", default=os.path.join("intermediate_result", "expert_activate"))
    parser.add_argument("--layers", type=int, nargs="+", default=[7])
    parser.add_argument("--include-random-layer", action="store_true")
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--bits", type=int, nargs="+", default=[0, 1, 2, 3, 4], choices=[0, 1, 2, 3, 4])
    parser.add_argument("--sort-bit", type=int, default=None, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--slices-per-expert", type=int, default=8)
    parser.add_argument("--target-bpw", type=float, default=2.0)
    parser.add_argument("--disable-0bit-compensation", action="store_true", default=True)
    parser.add_argument("--enable-0bit-compensation", dest="disable_0bit_compensation", action="store_false")
    parser.add_argument("--out-dir", default=os.path.join("figs", "dp_score_views"))
    args = parser.parse_args()
    args.views = ["assigned_loss_by_order"]

    model_id = resolve_model_id(args.model)
    cache_dir = resolve_cache_dir(args.cache_root, args.quantmode, args.rank_mode, model_id)
    layers = list(dict.fromkeys(args.layers))
    if args.include_random_layer:
        available_layers = discover_layers(cache_dir, model_id, args.bits[0])
        random_layer = choose_random_layer(available_layers, layers[0], args.random_seed)
        if random_layer not in layers:
            layers.append(random_layer)
    for layer in layers:
        run_for_layer(args, model_id, cache_dir, layer)


if __name__ == "__main__":
    main()
