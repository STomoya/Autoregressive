import torch
import torch.nn as nn
import torch.nn.functional as F

from autoregressive.config import ConfigMixin
from gatedpixelcnn.masked_conv import masked_conv2d
from pixelcnn.masked_conv import masked_conv2d as pixelcnn_masked_conv


class GatedResidual(nn.Module):
    def __init__(self, channels, kernel_size: int, mask_type: str = 'B'):
        super().__init__()

        self.v_conv = masked_conv2d(channels, channels * 2, kernel_size, mask_type=f'V{mask_type}')
        self.b_conv = nn.Conv2d(channels * 2, channels * 2, 1)
        self.h_conv = masked_conv2d(channels, channels * 2, kernel_size, mask_type=f'H{mask_type}')
        self.o_conv = nn.Conv2d(channels, channels, 1)

    def down_shift(self, x: torch.Tensor):
        x = x[:, :, :-1, :]
        x = F.pad(x, (0, 0, 1, 0), value=0.0)
        return x

    def forward(self, xs: tuple[torch.Tensor, torch.Tensor]):
        vx, hx = xs

        skip = hx

        vx = self.v_conv(vx)
        bx = self.b_conv(self.down_shift(vx))

        hx = self.h_conv(hx)

        hx = hx + bx

        hx_1, hx_2 = hx.chunk(2, dim=1)
        hx = torch.tanh(hx_1) * torch.sigmoid(hx_2)
        vx_1, vx_2 = vx.chunk(2, dim=1)
        vx = torch.tanh(vx_1) * torch.sigmoid(vx_2)

        hx = self.o_conv(hx)
        hx = skip + hx

        return (vx, hx)


class GatedPixelCNN(nn.Module, ConfigMixin):
    def __init__(
        self,
        num_layers: int = 8,
        channels: int = 64,
        kernel_size: int = 3,
        color_levels: int = 256,
        image_channels: int = 1,
    ):
        super().__init__()
        self.color_levels = color_levels

        self.input = pixelcnn_masked_conv(image_channels, channels, 7, mask_type='A', stride=1, padding=3)

        layers = []
        for _ in range(num_layers - 1):
            layers.append(
                GatedResidual(
                    channels,
                    kernel_size=kernel_size,
                )
            )
        self.net = nn.Sequential(*layers)

        self.output = pixelcnn_masked_conv(channels, color_levels * image_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        x = self.input(x)
        x = self.net((x, x))
        x = self.output(x[-1])  # we want the output of the hstack.
        x = x.reshape(B, self.color_levels, C, H, W)
        return x
