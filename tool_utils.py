import torch
import gc

def force_release_inactive_splits(device=0):
    # print(f" GPU {device} inactive_split_bytes...")

    gc.collect()
    torch.cuda.empty_cache()

    # remember and restore the original device: set_device changes the
    # process-wide current device; without restoring it, later triton kernels
    # launch in the wrong CUDA context (cuda:0 tensors + cuda:1 context ->
    # cuPointerGetAttribute INVALID_VALUE, reported as
    # "Pointer argument cannot be accessed from Triton (cpu tensor?)")
    orig_device = torch.cuda.current_device()
    torch.cuda.set_device(device)
    free_mem, total_mem = torch.cuda.mem_get_info()

    for ratio in [0.95, 0.9, 0.8, 0.7, 0.5, 0.3]:
        try:
            target_size = int(free_mem * ratio)
            dummy_tensor = torch.empty(target_size, dtype=torch.int8, device=f'cuda:{device}')
            del dummy_tensor
            break
        except:
            continue

    gc.collect()
    torch.cuda.empty_cache()

    torch.cuda.set_device(orig_device)
    
    # stats = torch.cuda.memory_stats(device=device)
    # print(f"inactive_split_bytes: {stats['inactive_split_bytes.all.current'] / 1024**3:.2f} GB")

def list_cuda_tensors(model, device):
    print(f"\n=== GPU {device} tensors ===")
    total = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda and obj.get_device() == device:
                sz = obj.numel() * obj.element_size()
                total += sz
                print(f"tensor: shape={obj.shape}, dtype={obj.dtype}, VRAM={sz/1024:.2f} KB")
        except:
            pass
    print(f"GPU {device} tensors VRAM: {total/1024**3:.2f} GB")

    total = 0
    
    for name, param in model.named_parameters():
        if param.is_cuda and param.get_device() == device:
            sz = param.numel() * param.element_size()
            total += sz
            print(f"parameter: {name}, shape={param.shape}, VRAM={sz/1024**2:.2f} MB")
    print(f"model parameters/buffers on GPU {device} VRAM: {total/1024**3:.2f} GB")
    
    total = 0
    for name, buf in model.named_buffers():
        if buf.is_cuda and buf.get_device() == device:
            sz = buf.numel() * buf.element_size()
            total += sz
            print(f"buffer: {name}, shape={buf.shape}, VRAM={sz/1024**2:.2f} MB")
    print(f"model buffers on GPU {device} VRAM: {total/1024**3:.2f} GB")

force_release_inactive_splits(device=0)