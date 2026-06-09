"""Legacy per-layer visualizations.

These functions were originally placed inside ``dp_utils.py`` (algorithm code)
and ``visual_utils.py``. They are kept here as a convenience re-export
collection for one-at-a-time debugging plots.  The *publication-quality*
figures live in ``viz.headroom`` and ``viz.metric_geometry``.

Usage
-----
    import viz.legacy  # or  from viz.legacy import plot_bit_overlap
    plot_bit_overlap(...)

Note that the actual function bodies live in ``dp_utils`` (the ``plot_*`` group)
and ``visual_utils`` (the Spearman / rank-scatter group); this module just
re-exports them under a clean namespace so that ``dp_utils`` can eventually
drop the plotting code without breaking legacy scripts.
"""

from dp_utils import (
    plot_bit_overlap,
    plot_block_losses_overlap,
    plot_neuron_rates_across_bits,
)
from visual_utils import (
    plot_diff_wbits_correlation,
    plot_spearman_rank_correlation,
)

__all__ = [
    "plot_neuron_rates_across_bits",
    "plot_bit_overlap",
    "plot_block_losses_overlap",
    "plot_diff_wbits_correlation",
    "plot_spearman_rank_correlation",
]