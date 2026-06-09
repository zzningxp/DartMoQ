"""
DartMoQ visualization package.

Layout
------
- _cache_io.py        : shared cache loading utilities (cross-model, cross-bit)
- headroom.py         : H.* — MoE mixed-precision has large untapped headroom
- metric_geometry.py  : G.* — different quantizers have intrinsically different
                        sensitivity geometries; element-wise MSE is degenerate for VQ
- distribution.py     : observation 2 — distributional differences across quantizers;
                        observation 3 — per-block log-loss is well-fit by log L(b)=p·b²+q·b+r
- legacy.py           : older one-layer-at-a-time visualizations (kept for reproducibility)

Each module exposes a `main()` that re-generates all the figures used in the paper.
The figures are written under `plot/headroom/` and `plot/metric_geometry/`.
"""
