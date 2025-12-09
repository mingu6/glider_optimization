import torch

def check_tensor_debug(x, name=None, print_stats=True):

    has_nan = torch.isnan(x).any().item()
    has_inf = torch.isinf(x).any().item()
    all_finite = not (has_nan or has_inf)

    if name is None:
        name = "Tensor"

    if print_stats:
        try:
            print(f"[DEBUG] {name}: shape={x.shape}, dtype={x.dtype}")
            print(f"    NaN: {has_nan}, Inf: {has_inf}, All finite: {all_finite}")
            if all_finite:
                print(f"    min={x.min().item():.6g}, max={x.max().item():.6g}, mean={x.mean().item():.6g}, std={x.std().item():.6g}")
        except Exception as e:
            print(f"[DEBUG] {name}: Could not compute stats: {e}")

    return {"has_nan": has_nan, "has_inf": has_inf, "all_finite": all_finite}