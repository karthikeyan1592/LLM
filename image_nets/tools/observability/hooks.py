def register_activation_hooks(model):
    """
    Attach one forward hook to each activation module in model.acts.
    The hook stores the output z(l) after every forward pass.

    Returns:
        store   : dict  {layer_idx -> z_tensor (CPU, detached)}
        handles : list  of hook handles — call h.remove() when done
    """
    store   = {}
    handles = []

    for i, act in enumerate(model.acts):
        def _hook(module, inp, out, idx=i):
            store[idx] = out.detach().cpu()
        handles.append(act.register_forward_hook(_hook))

    return store, handles


if __name__ == "__main__":
    import torch
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from tools.nets.mlp import MLP

    torch.manual_seed(0)
    model = MLP([3, 3, 2, 1])
    store, handles = register_activation_hooks(model)
    _ = model(torch.randn(10, 3))
    assert len(store) == 3, f"FAIL: expected 3 entries, got {len(store)}"
    assert store[0].shape == (10, 3), f"FAIL: L1 shape {store[0].shape}"
    assert store[1].shape == (10, 2), f"FAIL: L2 shape {store[1].shape}"
    assert store[2].shape == (10, 1), f"FAIL: L3 shape {store[2].shape}"
    for h in handles: h.remove()
    print("PASS: activation hooks OK")
