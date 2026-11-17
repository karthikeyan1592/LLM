import matplotlib.pyplot as plt


def plot_training_curves(history, save_path=None, title="Training curves"):
    """
    Plot train/val loss (top) and accuracy (bottom) from a list of history dicts.
    Each dict must have keys: epoch, train_loss, val_loss, train_acc, val_acc.
    """
    epochs = [e["epoch"]      for e in history]
    tr_l   = [e["train_loss"] for e in history]
    vl_l   = [e["val_loss"]   for e in history]
    tr_a   = [e["train_acc"]  for e in history]
    vl_a   = [e["val_acc"]    for e in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    ax1.plot(epochs, tr_l, lw=2, label="train")
    ax1.plot(epochs, vl_l, lw=2, linestyle="--", label="val")
    ax1.set_ylabel("BCE Loss"); ax1.legend(); ax1.grid(alpha=0.3)
    ax1.set_title(title)

    ax2.plot(epochs, tr_a, lw=2)
    ax2.plot(epochs, vl_a, lw=2, linestyle="--")
    ax2.set_ylabel("Accuracy"); ax2.set_xlabel("Epoch")
    ax2.set_ylim(0.4, 1.05); ax2.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()
    return fig
