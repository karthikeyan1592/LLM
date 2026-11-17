import torch


def layer_stats(model, act_store, lr):
    """
    Compute per-layer training diagnostics.
    Call AFTER loss.backward() and BEFORE optimizer.step().

    Returns a flat dict:
        L{l}_grad_norm    : Frobenius norm of weight gradient
        L{l}_weight_norm  : Frobenius norm of weight matrix
        L{l}_update_ratio : lr * grad_norm / weight_norm
        L{l}_act_std      : std of z(l) across the batch
        L{l}_sat_frac     : fraction of |z(l)| > 0.95  (tanh layers only; None for sigmoid)
    """
    entry = {}
    for l, lay in enumerate(model.layers):
        g_n = lay.weight.grad.norm().item() if lay.weight.grad is not None else 0.0
        w_n = lay.weight.norm().item()
        z   = act_store.get(l)
        is_tanh = l < len(model.layers) - 1

        entry[f"L{l+1}_grad_norm"]    = round(g_n, 6)
        entry[f"L{l+1}_weight_norm"]  = round(w_n, 6)
        entry[f"L{l+1}_update_ratio"] = round(lr * g_n / (w_n + 1e-8), 6)
        entry[f"L{l+1}_act_std"]      = round(z.std().item(), 6) if z is not None else None
        entry[f"L{l+1}_sat_frac"]     = (
            round((z.abs() > 0.95).float().mean().item(), 4)
            if (z is not None and is_tanh) else None
        )
    return entry


def gpu_stats(device):
    """
    Return GPU memory stats dict.
    Returns {} when device is CPU.
    """
    if not (hasattr(device, "type") and device.type == "cuda"):
        return {}
    return {
        "gpu_alloc_mb": round(torch.cuda.memory_allocated(device) / 1024 ** 2, 2),
        "gpu_peak_mb":  round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 2),
    }


if __name__ == "__main__":
    import torch, torch.nn.functional as F
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from tools.nets.mlp import MLP
    from tools.observability.hooks import register_activation_hooks

    torch.manual_seed(0)
    model = MLP([3, 3, 2, 1])
    store, handles = register_activation_hooks(model)
    x = torch.randn(16, 3); y = torch.randint(0, 2, (16, 1)).float()
    out  = model(x)
    loss = F.binary_cross_entropy(out, y)
    model.zero_grad(); loss.backward()
    stats = layer_stats(model, store, lr=0.1)
    assert "L1_grad_norm" in stats, "FAIL: missing key"
    assert stats["L3_sat_frac"] is None, "FAIL: sigmoid layer should have sat_frac=None"
    for h in handles: h.remove()
    print("PASS: diagnostics OK")
    print(stats)
