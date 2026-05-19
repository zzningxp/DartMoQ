# DartMoQ TurboQuant Backend

This file documents the local DartMoQ integration utilities in
`turboquant_model.dartmoq_backend`.

The integration uses TurboQuant only as the final fake-quant backend after
DartMoQ has already finished importance analysis, search, and mixed-bit
assignment.

It does not replace `nn.Linear` with `TurboQuantLinear`. The layer remains a
normal dense `nn.Linear`; only `linear.weight.data` is overwritten with the
dequantized TurboQuant approximation.

## What stays unchanged

- DartMoQ MoE reconstruction.
- `rank_mode=quant_outlier`, including GPTQ-based loss collection.
- Mixed-bit scheme parsing and search.
- 8-bit / 16-bit handling.

## What changes

For final selected 1-15 bit modules, call:

```python
from turboquant_model.dartmoq_backend import quantize_linear_if_turbo_supported

bit = gptq[name].quantizer.bits
result = quantize_linear_if_turbo_supported(
    gptq[name].layer,
    bit_width=bit,
    group_size=128,
    seed=42 + layer_idx,
    rotation="qr",
)

if not result.handled:
    loss[name] = gptq[name].fasterquant(
        name=f"layer_idx.{layer_idx}." + name,
        groupsize=groupsize,
        actorder=act_order,
        static_groups=static_groups,
    )
else:
    loss[name] = torch.zeros(1, device=gptq[name].layer.weight.device)
```

## Why rotation="qr"

TurboQuant's Hadamard rotation requires power-of-two group dimensions. DartMoQ
models and final groups may not always satisfy that, so this component defaults
to `rotation="qr"` for correctness. You can switch to `"hadamard"` later after
checking every quantized group size.

## Expected behavior

- bits 1-15: handled by TurboQuant fake-quant.
- bit 0: zeroes the weight by default, matching DartMoQ's current behavior.
- bit 16: treated as base/fp16 and left to the existing DartMoQ path.
- negative or invalid bit-widths: returns `handled=False`; keep the existing
  GPTQ/DartMoQ path or fail in the caller.

## Limitation

This is not real packed TurboQuant inference. It is for precision validation
and PPL comparison before moving to `TurboQuantLinear`.
