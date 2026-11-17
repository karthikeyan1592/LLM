import torch.nn as nn


class SmallCNN(nn.Module):
    """
    Conv(1→16,3×3)→ReLU→MaxPool → Conv(16→32,3×3)→ReLU→MaxPool
    → Flatten → FC(1568→128)→ReLU → FC(128→n_classes)

    Named submodules (conv1, conv2, relu3) make hook attachment explicit.
    """
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1  = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2  = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc1    = nn.Linear(32 * 7 * 7, 128)
        self.relu3  = nn.ReLU()
        self.fc2    = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)


if __name__ == "__main__":
    import torch
    m = SmallCNN()
    x = torch.randn(4, 1, 28, 28)
    out = m(x)
    assert out.shape == (4, 10), f"FAIL: {out.shape}"
    n = sum(p.numel() for p in m.parameters())
    print(f"PASS: output {out.shape}  params={n:,}")
