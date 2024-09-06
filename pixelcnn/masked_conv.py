import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parameterize


class MaskA(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()

        mask = torch.ones(1, 1, kernel_size, kernel_size)
        mask[..., kernel_size // 2, kernel_size // 2 :] = 0
        mask[..., kernel_size // 2 + 1 :, :] = 0

        self.register_buffer('mask', mask)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        masked = weight * self.mask
        return masked


class MaskB(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()

        mask = torch.ones(1, 1, kernel_size, kernel_size)
        mask[..., kernel_size // 2, kernel_size // 2 + 1 :] = 0
        mask[..., kernel_size // 2 + 1 :, :] = 0

        self.register_buffer('mask', mask)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        masked = weight * self.mask
        return masked


def masked_conv2d(in_channels, out_channels, kernel_size: int, mask_type: str = 'B', **kwargs):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, **kwargs)
    if mask_type == 'A':
        parameterize.register_parametrization(conv, 'weight', MaskA(kernel_size))
    elif mask_type == 'B':
        parameterize.register_parametrization(conv, 'weight', MaskB(kernel_size))
    return conv
