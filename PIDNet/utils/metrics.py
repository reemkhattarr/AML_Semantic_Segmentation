import torch
import time

def compute_flops_and_params(model: torch.nn.Module, input_shape=(1, 3, 1024, 1024)):
    """
    Computes FLOPs and parameter count for a model.
    Args:
        model: torch.nn.Module
        input_shape: (N, C, H, W)
    Returns:
        flops: number of multiply-adds (MACs)
        params: number of parameters
    """
    try:
        from thop import profile
        dummy = torch.randn(*input_shape).to(next(model.parameters()).device)
        flops, params = profile(model, inputs=(dummy,), verbose=False)
        return float(flops), float(params)
    except ImportError:
        print("Install thop for FLOPs/params computation: pip install thop")
        params = sum(p.numel() for p in model.parameters())
        return -1.0, float(params)

def measure_latency(model: torch.nn.Module, input_shape=(1, 3, 1024, 1024), num_runs=100, warmup=10, device='cuda'):
    """
    Measures model latencyA (ms) and FPS.
    """
    model.eval()
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model = model.to(device)
    dummy_input = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
    times = torch.tensor(times)
    mean_latency = times.mean().item() * 1000  # ms
    std_latency = times.std().item() * 1000
    fps = 1.0 / times.mean().item()
    return {
        "mean_latency_ms": mean_latency,
        "std_latency_ms": std_latency,
        "fps": fps
    }
