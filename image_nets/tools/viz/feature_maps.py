import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_feature_maps(model, image, device, save_path=None):
    """
    Run one image through the network and display every channel's activation
    after conv1 and conv2.

    Args:
        model  : SmallCNN (eval mode)
        image  : (1, 28, 28) float32 tensor — a single normalised image
        device : torch.device

    What you see: the spatial activation grid that each conv filter produces.
    Bright areas = where that filter responded strongly.
    """
    store = {}

    def make_hook(name):
        def hook(module, inp, out):
            store[name] = out.detach().cpu()
        return hook

    h1 = model.conv1.register_forward_hook(make_hook("conv1"))
    h2 = model.conv2.register_forward_hook(make_hook("conv2"))

    model.eval()
    with torch.no_grad():
        model(image.unsqueeze(0).to(device))

    h1.remove(); h2.remove()

    fig, big_axes = plt.subplots(2, 1, figsize=(16, 7))

    for ax_row, name in zip(big_axes, ["conv1", "conv2"]):
        acts = store[name][0]          # (C, H, W)
        C    = acts.shape[0]
        cols = min(C, 16)
        rows_inner = (C + cols - 1) // cols
        H, W = acts.shape[1], acts.shape[2]

        grid = np.zeros((rows_inner * (H + 1) - 1, cols * (W + 1) - 1))
        for k in range(C):
            r = k // cols; c = k % cols
            r0 = r * (H + 1); c0 = c * (W + 1)
            a = acts[k].numpy()
            a = (a - a.min()) / (a.max() - a.min() + 1e-8)
            grid[r0:r0 + H, c0:c0 + W] = a

        ax_row.imshow(grid, cmap="viridis", vmin=0, vmax=1)
        ax_row.set_title(f"{name}  —  {C} channels  "
                         f"({H}×{W} spatial)  [normalised 0–1 per channel]",
                         fontsize=10)
        ax_row.axis("off")

    fig.suptitle("Feature maps: what each filter activates for this image")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    return fig
