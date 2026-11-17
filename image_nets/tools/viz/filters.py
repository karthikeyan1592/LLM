import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_conv_filters(model, save_path=None):
    """
    Visualise the learned convolutional filters for conv1 and conv2.

    conv1 filters: shape (16, 1, 3, 3) — 16 grayscale 3×3 kernels.
    conv2 filters: shape (32, 16, 3, 3) — averaged over input channels
                   to give one representative 3×3 image per output channel.

    What you see: the spatial patterns each conv layer detects.
    Layer-1 filters are typically edge / corner / blob detectors.
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 5))

    for row, (name, module) in enumerate([("conv1", model.conv1[0]),
                                           ("conv2", model.conv2[0])]):
        W = module.weight.data.cpu().numpy()   # (out, in, kH, kW)
        n_out = W.shape[0]
        W_vis = W.mean(axis=1)                 # average over in-channels → (out, kH, kW)

        vmax = np.abs(W_vis).max()
        ax   = axes[row]
        ax.axis("off")
        ax.set_title(f"{name}  —  {n_out} filters  "
                     f"(kernel {W.shape[2]}×{W.shape[3]}, "
                     f"in-channels averaged)", fontsize=10)

        cols = min(n_out, 16)
        rows_inner = (n_out + cols - 1) // cols
        grid = np.ones((rows_inner * (W.shape[2] + 1) - 1,
                        cols * (W.shape[3] + 1) - 1)) * np.nan

        for k in range(n_out):
            r = k // cols
            c = k % cols
            r0 = r * (W.shape[2] + 1)
            c0 = c * (W.shape[3] + 1)
            grid[r0:r0 + W.shape[2], c0:c0 + W.shape[3]] = W_vis[k]

        ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  interpolation="nearest")

    fig.suptitle("Learned conv filters  (red = positive, blue = negative weight)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    return fig
