import torch
import torch.nn as nn

from autoregressive.activation import get_activation_cls
from autoregressive.config import ConfigMixin
from pixelcnn.masked_conv import masked_conv2d


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, activation: str, reduction: int = 2, bn: bool = True):
        super().__init__()

        inner_channels = channels // reduction

        self.bn1 = nn.BatchNorm2d(channels) if bn else nn.Identity()
        self.conv1 = nn.Conv2d(channels, inner_channels, 1)
        self.bn2 = nn.BatchNorm2d(inner_channels) if bn else nn.Identity()
        self.conv2 = masked_conv2d(inner_channels, inner_channels, kernel_size, stride=1, padding=kernel_size // 2)
        self.bn3 = nn.BatchNorm2d(inner_channels) if bn else nn.Identity()
        self.conv3 = nn.Conv2d(inner_channels, channels, 1)
        self.act = get_activation_cls(activation)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.conv3(x)
        x = x + skip
        return x


class PixelCNN(nn.Module, ConfigMixin):
    def __init__(
        self,
        num_layers: int = 8,
        channels: int = 64,
        kernel_size: int = 3,
        color_levels: int = 256,
        image_channels: int = 1,
        activation: str = 'relu',
    ):
        super().__init__()
        self.color_levels = color_levels

        layers = [
            masked_conv2d(image_channels, channels, 7, mask_type='A', stride=1, padding=3),
            nn.BatchNorm2d(channels),
            get_activation_cls(activation)(),
        ]

        for _ in range(num_layers - 1):
            layers.append(
                ResidualBlock(
                    channels,
                    kernel_size=kernel_size,
                    activation=activation,
                )
            )

        layers.append(masked_conv2d(channels, color_levels * image_channels, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        x = self.net(x)
        x = x.reshape(B, self.color_levels, C, H, W)
        return x


@torch.no_grad()
def sample_image(
    model: PixelCNN, size: tuple[int, int, int, int], device: torch.device, normalize: bool = True
) -> torch.Tensor:
    model.eval()
    _, C, H, W = size
    images = torch.zeros(size, device=device)
    # DDP stuff.
    color_levels = model.color_levels if hasattr(model, 'color_levels') else model.module.color_levels

    for h in range(H):
        for w in range(W):
            for c in range(C):
                logits = model(images)
                probs = logits[:, :, c, h, w].softmax(1)
                level = torch.multinomial(probs, 1).squeeze().float() / (color_levels - 1)
                pixel_value = level * 2 - 1 if normalize else level
                images[:, c, h, w] = pixel_value

    model.train()
    return images
