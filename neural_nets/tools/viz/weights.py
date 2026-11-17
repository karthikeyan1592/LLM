import numpy as np
import matplotlib.pyplot as plt


def plot_weight_heatmaps(weight_snaps, epochs, save_path=None):
    """
    Show weight matrices for each layer at selected epochs + the change.

    Args:
        weight_snaps : dict  {epoch -> {"L1_W": np.array, "L1_b": np.array, ...}}
        epochs       : list  of epoch keys to show, e.g. [0, 150, 300]
        save_path    : optional path to save the figure
    """
    first_epoch = epochs[0]
    last_epoch  = epochs[-1]
    snap0       = weight_snaps[first_epoch]
    snap_last   = weight_snaps[last_epoch]

    layer_keys  = sorted(k for k in snap0 if k.endswith("_W"))
    n_layers    = len(layer_keys)

    fig, axes = plt.subplots(n_layers, 3, figsize=(11, 3.5 * n_layers))
    if n_layers == 1:
        axes = axes[None, :]

    for row, key in enumerate(layer_keys):
        label   = key.replace("_W", "")
        W_before = snap0[key]
        W_after  = snap_last[key]
        W_delta  = W_after - W_before
        vmax = max(abs(W_before).max(), abs(W_after).max(), abs(W_delta).max()) + 1e-6

        for col, (mat, title) in enumerate([
            (W_before, f"{label} W — epoch {first_epoch}"),
            (W_after,  f"{label} W — epoch {last_epoch}"),
            (W_delta,  f"{label} ΔW  (after − before)"),
        ]):
            ax = axes[row, col]
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("input neuron"); ax.set_ylabel("output neuron")
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            fontsize=7, color="k")
            plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Weight matrices: before, after training, and the change", y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()
    return fig
