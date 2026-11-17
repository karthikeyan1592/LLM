import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def gradcam(model, image, device):
    """
    Grad-CAM on the conv2 block (last conv layer).

    Algorithm:
      1. Forward pass — capture conv2 feature maps.
      2. Backward on the predicted class score — capture conv2 gradients.
      3. Global-average-pool the gradients → per-channel weights.
      4. Weighted sum of feature maps → raw CAM.
      5. ReLU → resize to input spatial size.

    Args:
        model  : SmallCNN
        image  : (1, 28, 28) float32 tensor
        device : torch.device

    Returns:
        cam  : (28, 28) numpy array in [0, 1]
        pred : predicted class index (int)
    """
    model.eval()
    features, grads = {}, {}

    fwd_h = model.conv2.register_forward_hook(
        lambda m, i, o: features.update({"conv2": o})
    )
    bwd_h = model.conv2.register_full_backward_hook(
        lambda m, gi, go: grads.update({"conv2": go[0]})
    )

    x   = image.unsqueeze(0).to(device).requires_grad_(False)
    out = model(x)
    pred = int(out.argmax(1))

    model.zero_grad()
    out[0, pred].backward()

    fwd_h.remove(); bwd_h.remove()

    feat    = features["conv2"][0].detach()   # (C, H, W)
    grad    = grads["conv2"][0].detach()      # (C, H, W)
    weights = grad.mean(dim=(1, 2))           # (C,)

    cam = (weights[:, None, None] * feat).sum(0)
    cam = F.relu(cam)
    cam = cam / (cam.max() + 1e-8)

    cam_np = cam.cpu().numpy()
    cam_28 = F.interpolate(
        torch.tensor(cam_np)[None, None],
        size=(28, 28), mode="bilinear", align_corners=False
    )[0, 0].numpy()

    return cam_28, pred


def plot_gradcam_grid(model, images, labels, device,
                      class_names=None, n=8, save_path=None):
    """
    Plot n examples: original image | Grad-CAM heatmap | overlay.

    Args:
        model       : SmallCNN
        images      : (N, 1, 28, 28) tensor
        labels      : (N,) int tensor
        n           : how many examples to show
        class_names : list of 10 strings; defaults to digit strings
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]

    fig, axes = plt.subplots(n, 3, figsize=(7, 2.5 * n))

    for row in range(n):
        img = images[row]                    # (1, 28, 28)
        cam, pred = gradcam(model, img, device)

        raw   = img[0].numpy()
        raw_n = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)

        axes[row, 0].imshow(raw_n, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"true: {class_names[int(labels[row])]}", fontsize=8)

        axes[row, 1].imshow(cam, cmap="jet", vmin=0, vmax=1)
        axes[row, 1].set_title(f"Grad-CAM → pred: {class_names[pred]}", fontsize=8)

        axes[row, 2].imshow(raw_n, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(cam,   cmap="jet",  vmin=0, vmax=1, alpha=0.45)
        axes[row, 2].set_title("overlay", fontsize=8)

        for ax in axes[row]:
            ax.axis("off")

    col_titles = ["Input", "Grad-CAM", "Overlay"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=9, fontweight="bold")

    fig.suptitle("Grad-CAM — which pixels drove each prediction")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    return fig
