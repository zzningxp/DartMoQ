import torch
import torch.nn as nn
import transformers
from data_utils import *
import time

def simple_sft(model, tokenizer, args, epoch = 1):
    nsample = args.nsamples
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=model.seqlen, bsz = args.sft_bsz
    )

    # Set up gate-specific training parameters
    flag = False
    for layer in model.model.layers:
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
            flag = True
    if not flag:
        print("No gate found in the model.")
        return model


    # Separate parameters for optimization - only train gate-related parameters

    for name, param in model.named_parameters():
        if 'gate.gate.weight' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False  # Freeze other parameters

    lr = 2e-3
    wd = 1e-2
    betas = (0.9, 0.95)
    eps = 1e-8

    # Only optimize gate parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=wd
    )

    # print(len(gate_params), len(dataloader))
    print("trainable count:", sum(1 for p in model.parameters() if p.requires_grad))
    print(model.model.layers[0].mlp.gate.gate.weight)
    print(model.model.layers[-1].mlp.gate.gate.weight)
    # print(model.model.layers[0].mlp.experts[0].gate_proj.weight[0])
    # print(model.model.layers[-1].mlp.experts[0].gate_proj.weight[0])

    num_epoch=epoch
    model.train()
    for epoch in range(num_epoch):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(batch[0].to('cuda'), labels = batch[0].to('cuda'))
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    
            # for name, param in model.named_parameters():
            #     if param.requires_grad and param.grad is not None:
            #         grad_norm = torch.norm(param.grad)
            #         print(f"{name} - grad norm: {grad_norm.item()}")
        avg_loss = epoch_loss/len(dataloader)
        print(f'epoch:{epoch}, avg_loss:{avg_loss}')
        print(model.model.layers[0].mlp.gate.gate.weight)
        print(model.model.layers[-1].mlp.gate.gate.weight)
        # print(model.model.layers[0].mlp.experts[0].gate_proj.weight[0])
        # print(model.model.layers[-1].mlp.experts[0].gate_proj.weight[0])

    return model