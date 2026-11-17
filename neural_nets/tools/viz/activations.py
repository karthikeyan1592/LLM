import numpy as np
import matplotlib.pyplot as plt
import torch

from ..observability.hooks import register_activation_hooks


def plot_activation_landscape(model, device, layer_idx=0, n_grid=60, save_path=None):
    """
    Sweep a 2D grid of (x1, x2) with x3=0 and show each neuron's activation
    as a filled contour map.

    What you see: the region of input space where each neuron fires strongly
    (red = high activation, blue = low/negative).

    Args:
        model     : trained MLP
        device    : torch.device
        layer_idx : which layer to inspect (0=L1, 1=L2, ...)
        n_grid    : grid resolution per axis
        save_path : optional save path
    """
    model.eval()
    xs = np.linspace(-3, 3, n_grid)
    ys = np.linspace(-3, 3, n_grid)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.stack([XX.ravel(), YY.ravel(),
                     np.zeros(n_grid * n_grid)], axis=1).astype(np.float32)

    store, handles = register_activation_hooks(model)
    with torch.no_grad():
        model(torch.tensor(grid).to(device))
    for h in handles:
        h.remove()

    Z         = store[layer_idx].numpy()      # (n_grid², n_neurons)
    n_neurons = Z.shape[1]

    fig, axes = plt.subplots(1, n_neurons, figsize=(4 * n_neurons, 4))
    if n_neurons == 1:
        axes = [axes]
    for k, ax in enumerate(axes):
        vals = Z[:, k].reshape(n_grid, n_grid)
        vmax = abs(vals).max()
        cf = ax.contourf(XX, YY, vals, levels=20, cmap="RdBu_r",
                         vmin=-vmax, vmax=vmax)
        ax.contour(XX, YY, vals, levels=[0], colors="k", linewidths=1)
        ax.set_title(f"Layer {layer_idx + 1}, Neuron {k + 1}\n(x₃ = 0)", fontsize=9)
        ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
        plt.colorbar(cf, ax=ax, fraction=0.046)

    fig.suptitle(f"Activation landscape — what activates each neuron in layer {layer_idx + 1}")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()
    return fig


def plot_layer_representations(model, X, y, device, save_path=None):
    """
    For each layer, scatter the layer's output coloured by true class label.
    Shows how the network 'untangles' the non-linear boundary layer by layer.

    Input (raw features) is shown first.
    Layers with >2 neurons use the first two dimensions.
    The output layer (1 neuron) is shown as a 1D strip.

    Args:
        model : trained MLP
        X     : float32 numpy array (N, n_features)
        y     : int numpy array    (N,)
        device: torch.device
    """
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)

    store, handles = register_activation_hooks(model)
    with torch.no_grad():
        model(X_t)
    for h in handles:
        h.remove()

    # build list of (title, 2-column array)
    panels = [("Input (x₁ vs x₂)", X[:, :2])]
    for l in range(len(model.layers)):
        z = store[l].numpy()
        label = f"After L{l + 1}"
        if z.shape[1] >= 2:
            panels.append((label, z[:, :2]))
        else:
            # 1-neuron output: show as strip (prediction vs index)
            idx = np.arange(len(z))
            panels.append((label + " (output)", np.stack([idx, z[:, 0]], axis=1)))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    colors = ["C0", "C1"]

    for ax, (title, pts) in zip(axes, panels):
        for cls in [0, 1]:
            mask = y == cls
            ax.scatter(pts[mask, 0], pts[mask, 1], s=8, alpha=0.5,
                       color=colors[cls], label=f"class {cls}")
        ax.set_title(title, fontsize=9)
        ax.legend(markerscale=2, fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Layer-by-layer representation — the network untangling the boundary")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()
    return fig


def plot_neuron_weights(model, save_path=None, feature_names=None):
    """
    Bar chart of W(1) weight vectors — which input features each L1 neuron
    amplifies or suppresses.

    Args:
        model         : trained MLP
        feature_names : list of strings for x-axis labels
    """
    W1 = model.layers[0].weight.data.cpu().numpy()   # (n_L1, n_features)
    n_neurons, n_features = W1.shape
    if feature_names is None:
        feature_names = [f"x{i+1}" for i in range(n_features)]

    fig, axes = plt.subplots(1, n_neurons, figsize=(3.5 * n_neurons, 3.5))
    if n_neurons == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        w = W1[k]
        colors = ["C3" if v > 0 else "C0" for v in w]
        ax.bar(feature_names, w, color=colors)
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_title(f"L1 Neuron {k + 1}", fontsize=9)
        ax.set_ylabel("weight")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Layer 1 weight vectors — which features each neuron responds to\n"
                 "(red = amplify, blue = suppress)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()
    return fig
