
if __name__ == '__main__':
    import sys
    import os
    file_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(file_dir)
    sys.path.insert(0, parent_dir)
    sys.path.insert(0, file_dir)

    import argparse
    import torch
    import torch.nn as nn
    from eval_dartmoq import load_model

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Model to load')
    parser.add_argument('--wbits', type=int, default=4, help='Bit width for quantization')
    parser.add_argument('--group-size', type=int, default=128, help='Group size for quantization')
    parser.add_argument('--test-update-false', action='store_true', help='Also test update=False mode')
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model)
    model.eval()

    # Directly get layer 1, expert 0, up_proj
    layer = model.model.layers[1]
    linear = layer.mlp.experts[0].up_proj

    print(f"\nTesting layer 1, expert 0, up_proj:")
    print(f"  Weight shape: {linear.weight.shape}")
    print(f"  Weight dtype: {linear.weight.dtype}")
    print(f"  Weight device: {linear.weight.device}")

    # Test normal mode (update=True)
    print(f"\n--- Testing update=True mode ---")
    weight_orig = linear.weight.data.clone()

    quant_error = turbo_fake_quant_linear(
        linear,
        bit_width=args.wbits,
        group_size=args.group_size,
        seed=42,
        rotation='qr',
        update=True
    )

    print(f"Quant error sum: {quant_error.sum().item():.6f}")
    print(f"Quant error mean: {quant_error.mean().item():.6f}")
    print(f"Quant error max: {quant_error.max().item():.6f}")

    # Verify weight was updated
    weight_diff = (linear.weight.data - weight_orig).abs().sum()
    print(f"Weight updated, sum of absolute differences: {weight_diff.item():.6f}")

    # Restore original weight
    linear.weight.data.copy_(weight_orig)

    # Test update=False mode if requested
    if args.test_update_false:
        print(f"\n--- Testing update=False mode ---")
        neuron_direction = 'up'
        print(f"Using neuron_direction: {neuron_direction}")

        quant_error_per_neuron = turbo_fake_quant_linear(
            linear,
            bit_width=args.wbits,
            group_size=args.group_size,
            seed=42,
            rotation='qr',
            update=False,
            neuron_direction=neuron_direction
        )

        print(f"Quant error shape: {quant_error_per_neuron.shape}")
        print(f"Quant error sum: {quant_error_per_neuron.sum().item():.6f}")
        print(f"Quant error mean: {quant_error_per_neuron.mean().item():.6f}")
        print(f"Quant error max: {quant_error_per_neuron.max().item():.6f}")

        # Verify weight was NOT updated
        weight_diff = (linear.weight.data - weight_orig).abs().sum()
        print(f"Weight NOT updated, sum of absolute differences: {weight_diff.item():.6f}")

    print("\nTest completed!")

