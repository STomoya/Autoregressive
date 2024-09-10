import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parameterize


class Mask(nn.Module):
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        weight = weight * self.mask
        return weight

    def register_mask(self, mask: torch.Tensor):
        self.register_buffer('mask', mask)


class MaskVA(Mask):
    def __init__(self, kernel_size: int):
        super().__init__()
        mask = torch.ones(1, 1, kernel_size, kernel_size)
        mask[:, :, kernel_size // 2, kernel_size // 2 :] = 0.0
        mask[:, :, kernel_size // 2 + 1 :, :] = 0.0
        self.register_mask(mask)


class MaskHA(Mask):
    def __init__(self, kernel_size: int):
        super().__init__()
        mask = torch.ones(1, 1, 1, kernel_size)
        mask[:, :, :, kernel_size // 2 :] = 0.0
        self.register_mask(mask)


class MaskVB(Mask):
    def __init__(self, kernel_size: int):
        super().__init__()
        mask = torch.ones(1, 1, kernel_size, kernel_size)
        mask[:, :, kernel_size // 2, kernel_size // 2 + 1 :] = 0.0
        mask[:, :, kernel_size // 2 + 1 :, :] = 0.0
        self.register_mask(mask)


class MaskHB(Mask):
    def __init__(self, kernel_size: int):
        super().__init__()
        mask = torch.ones(1, 1, 1, kernel_size)
        mask[:, :, :, kernel_size // 2 + 1 :] = 0.0
        self.register_mask(mask)


MASK_TYPE_TO_CLS = {
    'HA': MaskHA,
    'HB': MaskHB,
    'VA': MaskVA,
    'VB': MaskVB,
}


def masked_conv2d(in_channels, out_channels, kernel_size: int, mask_type: str = 'VB', **kwargs):
    padding = kwargs.pop('padding', kernel_size // 2)
    if mask_type.startswith('H'):
        kernel_size = (1, kernel_size)
        padding = (0, padding)

    conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, **kwargs)
    mask = MASK_TYPE_TO_CLS[mask_type](kernel_size if isinstance(kernel_size, int) else kernel_size[-1])

    return parameterize.register_parametrization(conv, 'weight', mask)
