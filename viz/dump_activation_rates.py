"""Dump per-layer expert activation rates to ``intermediate_result/expert_activate/{model_id}/``.

H.3 (activation × sensitivity correlation, see ``viz/headroom.py``) needs the
per-layer expert activation rate measured on the calibration set. The main
quantization pipeline computes this internally inside
:func:`dartmoq_utils.analyze_experts_activation` but does not persist it; this
dumper runs the same forward pass once and writes one ``Lx.npy`` per decoder
layer.

It deliberately mirrors the activation-capture pattern of
``dartmoq_sequential.dartmoq_sequential`` (``Catcher`` on layer 0 → propagate
hidden states layer-by-layer) so that the recorded rates are identical to what
the main pipeline sees.

Usage
-----
    python -m viz.dump_activation_rates --model /home/daodao/models/OLMoE-1B-7B-0924-Instruct/
    python -m viz.dump_activation_rates --model olmoe-7b-1b --nsamples 64

If the output file already exists for a layer, that layer is skipped. Delete
the ``intermediate_result/expert_activate/<model_id>/`` directory to force a re-dump.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

# Make sibling modules importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import get_loaders
from dartmoq_utils import analyze_experts_activation
from eval_dartmoq import load_model
from viz._cache_io import resolve_model_id, resolve_model_path

DEV = torch.device("cuda:0")


@torch.no_grad()
def dump(
    model_path_or_id: str,
    nsamples: int = 64,
    seqlen: int = 2048,
    out_root: str = "intermediate_result/expert_activate",
    seed: int = 0,
    overwrite: bool = False,
) -> str:
    """Run one calibration pass and dump per-layer activation rates."""
    # Accept either a short cache id ("olmoe-7b-1b") or a real model path;
    # convert to a real path before handing to load_model().
    model_path = resolve_model_path(model_path_or_id)
    print(f"loading model {model_path}")
    model, tokenizer = load_model(model_path)
    model.seqlen = seqlen
    short_id = resolve_model_id(getattr(model, "model_id", model_path))
    model.model_id = short_id

    out_dir = os.path.join(out_root, short_id)
    os.makedirs(out_dir, exist_ok=True)
    print(f"writing to {out_dir}")

    dataloader, _ = get_loaders(
        "wikitext2", nsamples=nsamples, seed=seed,
        seqlen=seqlen, tokenizer=tokenizer, bsz=1,
    )

    # ----- run layer-0 catcher to obtain (inps, attention_mask, position_*) -----
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    dtype = next(iter(model.parameters())).dtype

    inps = torch.zeros((nsamples, 1, seqlen, model.config.hidden_size),
                       dtype=dtype, device="cpu")
    cache = {"i": 0, "attention_mask": None, "position_ids": None,
             "position_embeddings": None}

    class Catcher(nn.Module):
        def __init__(self, module): super().__init__(); self.module = module
        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            cache["position_embeddings"] = kwargs.get("position_embeddings")
            raise ValueError
        def __getattr__(self, name):
            try: return super().__getattr__(name)
            except AttributeError: return getattr(self.module, name)

    model.model.embed_tokens = model.model.embed_tokens.to(DEV)
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(DEV))
        except ValueError:
            pass
        if cache["i"] >= nsamples:
            break
    layers[0] = layers[0].module
    torch.cuda.empty_cache()
    inps = inps.squeeze(1)  # (nsamples, seqlen, hidden)

    # ----- iterate every layer; compute and persist activation rate ----------
    modeltype = model.config.model_type
    if hasattr(model.config, "num_experts_per_tok"):
        top_k = model.config.num_experts_per_tok
    else:
        top_k = 1

    n_layers = len(layers)
    print(f"capturing {n_layers} layers, top_k={top_k}, modeltype={modeltype}")

    new_inps = torch.zeros_like(inps)
    for layer_idx, layer in enumerate(layers):
        out_path = os.path.join(out_dir, f"{short_id}_L{layer_idx}.pt")
        skip = os.path.exists(out_path) and not overwrite

        dev = next(layer.parameters()).device
        attention_mask = cache["attention_mask"].to(dev) if cache["attention_mask"] is not None else None
        position_ids = cache["position_ids"].to(dev) if cache["position_ids"] is not None else None
        pe = cache["position_embeddings"]

        # ---- forward to obtain hidden_states going into the MLP block ------
        for b_i in range(inps.shape[0]):
            inp_b = inps[b_i:b_i+1].to(dev)
            residual = inp_b
            h = layer.input_layernorm(inp_b)
            if modeltype in ("olmoe", "llama", "qwen3", "qwen3_moe", "deepseek_v3"):
                attn = layer.self_attn(
                    hidden_states=h, attention_mask=attention_mask,
                    position_ids=position_ids, position_embeddings=pe)[0]
            else:
                attn = layer.self_attn(
                    hidden_states=h, attention_mask=attention_mask,
                    position_ids=position_ids)[0]
            h = residual + attn
            residual = h
            h = layer.post_attention_layernorm(h)

            if b_i == 0:
                h_all = torch.zeros((inps.shape[0], h.shape[1], h.shape[2]),
                                    dtype=h.dtype, device=dev)
            h_all[b_i] = h.squeeze(0)

            # finish forward through MLP for the *next* layer's input
            mlp_out = layer.mlp(h)
            mlp_out = mlp_out[0] if isinstance(mlp_out, tuple) else mlp_out
            new_inps[b_i] = (mlp_out + residual).squeeze(0).cpu()
            del inp_b, h, attn, mlp_out, residual

        # ---- compute activation rate from the captured pre-MLP hidden_states
        is_moe = hasattr(layer.mlp, "gate") or hasattr(layer.mlp, "experts")
        if skip:
            print(f"  L{layer_idx}: already exists, skipping")
        elif is_moe:
            try:
                rates = analyze_experts_activation(
                    layer, layer_idx, h_all, top_k, modeltype, save_path=None,
                )
                torch.save(rates.detach().cpu(), out_path)
                print(f"  L{layer_idx}: saved {out_path}  (shape={tuple(rates.shape)})")
            except Exception as e:
                print(f"  L{layer_idx}: failed ({e}); skipping")
        else:
            print(f"  L{layer_idx}: dense MLP (no gate), skipping")

        inps = new_inps.clone()
        del h_all
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    print(f"\ndone. activation rates written under {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="short cache id OR full model path (e.g. /home/daodao/models/OLMoE-1B-7B-0924/)")
    parser.add_argument("--nsamples", type=int, default=64)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-root", default="intermediate_result/expert_activate")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-dump even if layer .pt already exists")
    args = parser.parse_args()
    dump(args.model, args.nsamples, args.seqlen,
         out_root=args.out_root, seed=args.seed, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
