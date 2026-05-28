import os
import json
import torch


def save_dartmoq_model(model, tokenizer, save_dir, args=None):
    """
    Save DartMoQ model with proper handling of DartMoQHybridWrapper.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save model weights using Hugging Face format
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Save additional metadata for reconstruction
    metadata = {
        "model_type": getattr(model.config, "model_type", None),
        "use_hybrid_moe": getattr(args, "use_hybrid_moe", True) if args else True,
    }
    if args:
        metadata.update({
            "rank_mode": getattr(args, "rank_mode", None),
            "quant_scheme": getattr(args, "quant_scheme", None),
            "slices": getattr(args, "slices", None),
        })

    with open(os.path.join(save_dir, "dartmoq_config.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"DartMoQ model saved to {save_dir}")


def load_dartmoq_model(save_dir, trust_remote_code=True, device_map="auto"):
    """
    Load DartMoQ model with DartMoQHybridWrapper support.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Ensure DartMoQHybridWrapper is in the global scope for unpickling
    import sys
    import dartmoq_hybridmoe
    sys.modules['dartmoq_hybridmoe'] = dartmoq_hybridmoe

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(save_dir, trust_remote_code=trust_remote_code)

    # Load model - DartMoQHybridWrapper will be loaded automatically as it's a nn.Module
    model = AutoModelForCausalLM.from_pretrained(
        save_dir,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        torch_dtype="auto",
    )

    # Load and store dartmoq config if available
    config_path = os.path.join(save_dir, "dartmoq_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            model.dartmoq_config = json.load(f)

    # Set seqlen if not already set
    if not hasattr(model, "seqlen"):
        model.seqlen = 2048

    # Set model_id if not already set
    if not hasattr(model, "model_id"):
        model.model_id = os.path.basename(save_dir)

    return model, tokenizer
