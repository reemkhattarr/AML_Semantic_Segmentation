# measure_utils.py

import time, torch
from contextlib import nullcontext

def count_params(model, trainable_only=True):
    params = (p for p in model.parameters() if (p.requires_grad or not trainable_only))
    total = sum(p.numel() for p in params)
    return total, f"{total/1e6:.2f}M"

@torch.no_grad()
def measure_latency(model, 
                    input_size=(1, 3, 512, 512),
                    device="cuda",
                    warmup=10,
                    iters=50,
                    amp=False):
    model = model.to(device).eval()
    x = torch.randn(*input_size, device=device)
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (amp and device=="cuda") else nullcontext()

    for _ in range(warmup):
        with amp_ctx:
            _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with amp_ctx:
            _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    avg = sum(times) / len(times)
    p50 = sorted(times)[len(times)//2]
    p95 = sorted(times)[int(len(times)*0.95)-1]
    return {
        "avg_ms": avg,
        "p50_ms": p50,
        "p95_ms": p95,
        "throughput_fps": 1000.0 / avg if avg > 0 else float("inf")
    }

def count_flops(model, input_size=(1, 3, 512, 512), device="cpu"):
    model = model.to(device).eval()
    x = torch.randn(*input_size, device=device)

    with torch.no_grad():
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU] + (
                [torch.profiler.ProfilerActivity.CUDA] if device=="cuda" else []
            ),
            with_flops=True
        ) as prof:
            _ = model(x)

    flops = 0
    for evt in prof.key_averages():
        if hasattr(evt, "flops") and evt.flops is not None:
            flops += evt.flops

    return flops / 1e9, f"{flops/1e9:.2f} GFLOPs"
