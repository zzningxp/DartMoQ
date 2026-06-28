import torch
import torch.nn as nn
import torch.nn.functional as F

class DartMoQHybridWrapper(nn.Module):
    """
    Single expert wrapper for hybrid MoE.
    This class wraps multiple sub-experts (with different bit configs) into a single callable expert.
    When called, it forwards input through all sub-experts and sums the results.
    """
    def __init__(self, sub_experts):
        super().__init__()
        # Wrap each expert's sub-experts in a ModuleList
        self.sub_experts = sub_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # If all sub-experts are pruned (0bit), return zero directly
        if len(self.sub_experts) == 0:
            return torch.zeros_like(hidden_states)

        # Fast path for single sub-expert (common case from logs)
        if len(self.sub_experts) == 1:
            return self.sub_experts[0](hidden_states)

        # For multiple sub-experts, accumulate outputs
        total_output = self.sub_experts[0](hidden_states)
        for sub_expert in self.sub_experts[1:]:
            total_output = total_output + sub_expert(hidden_states)

        return total_output

    def named_children(self):
        """
        Override named_children to directly return sub-experts.
        This allows find_layers to properly discover the parameters in sub-experts.
        """
        for i, sub_expert in enumerate(self.sub_experts):
            yield (f"sub_expert_{i}", sub_expert)

    def __getitem__(self, idx):
        """Allow indexing like a list to get sub-experts of an expert."""
        return self.sub_experts[idx]

    def __len__(self):
        """Return number of sub-experts."""
        return len(self.sub_experts)

def restructure_hybrid_qscheme(qscheme_expert, slice_expert_num):

    restructured = []
    for expert_idx in range(len(qscheme_expert)):
        # Count occurrences of each bit config
        bit_counts = {}
        for bit in qscheme_expert[expert_idx]:
            bit_counts[bit] = bit_counts.get(bit, 0) + 1

        # Create list of unique bits, sorted by bit value (descending), consistent with sub-expert creation order
        # Keep bit == 0 in qscheme for quantization phase lookup, but we will skip creating them
        expert_bits = sorted(bit_counts.items(), reverse=True)
        restructured.append([bit for bit, count in expert_bits])

        # print(f"Expert {expert_idx} original: {qscheme_expert[expert_idx]} -> restructured: {restructured[expert_idx]} {expert_bits}")

    return restructured