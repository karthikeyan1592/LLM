import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Configurable MLP with explicit activation modules (required for forward hooks).

    Notation (matches user notes):
        W(l) shape: (neurons_l, neurons_{l-1})   <- same as nn.Linear.weight
        a(l) = W(l) · z(l-1) + b(l)              pre-activation
        z(l) = activation(a(l))                   post-activation
        delta(l) = dE/da(l)                       error signal

    Args:
        sizes  : list of layer widths, e.g. [3, 3, 2, 1]
        output : "sigmoid" (binary) or "linear" (use with CrossEntropyLoss)
    """
    def __init__(self, sizes, output="sigmoid"):
        super().__init__()
        self.sizes = sizes
        self.layers = nn.ModuleList([
            nn.Linear(sizes[i], sizes[i + 1])
            for i in range(len(sizes) - 1)
        ])
        if output == "sigmoid":
            out_act = nn.Sigmoid()
        else:
            out_act = nn.Identity()

        self.acts = nn.ModuleList(
            [nn.Tanh() for _ in range(len(sizes) - 2)] + [out_act]
        )

    def forward(self, x):
        z = x
        for linear, act in zip(self.layers, self.acts):
            z = act(linear(z))          # a = W·z + b;  z = act(a)
        return z


if __name__ == "__main__":
    import torch
    torch.manual_seed(0)
    net = MLP([3, 3, 2, 1])
    x   = torch.randn(5, 3)
    out = net(x)
    assert out.shape == (5, 1), f"FAIL: {out.shape}"
    assert out.min() >= 0 and out.max() <= 1, "FAIL: sigmoid range"
    print("PASS: MLP forward OK")
