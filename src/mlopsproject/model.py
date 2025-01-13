from torch import nn
import torch
import torch.nn.functional as F


class MyAwesomeModel(nn.Module):
    """Convolutional neural network for MNIST classification."""

    def __init__(self) -> None:
        super().__init__()
        # Input is 28x28 images, so convolution layer will have 1 input channel
        # Output is 10 classes, so output layer will have 10 outputs
        self.conv1_1 = nn.Conv2d(1, 10, 3, padding=1)
        self.conv1_2 = nn.Conv2d(10, 10, 3, padding=1)

        self.conv2_1 = nn.Conv2d(10, 10, 3, padding=1)
        self.conv2_2 = nn.Conv2d(10, 10, 3, padding=1)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # Fully connected layer: 10x7x7 flattened to 490 → 10 classes
        self.fc = nn.Linear(10 * 7 * 7, 10)

    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1_1(x))  # 28x28x1 → 26x26x10
        x = F.relu(self.conv1_2(x))  # 26x26x10 → 24x24x10
        x = self.maxpool(x)  # 24x24x10 → 12x12x10

        # Block 2
        x = F.relu(self.conv2_1(x))  # 12x12x10 → 10x10x10
        x = F.relu(self.conv2_2(x))  # 10x10x10 → 8x8x10
        x = self.maxpool(x)  # 8x8x10 → 4x4x10

        # Flattening
        x = torch.flatten(x, start_dim=1)  # 4x4x10 → 160 # Preserve batch dimension

        # Fully connected layer
        x = self.dropout(x)
        x = self.fc(x)  # 160 → 10 classes

        return x


if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
