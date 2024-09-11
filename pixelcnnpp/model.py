import torch
import torch.nn as nn
import torch.nn.functional as F

from autoregressive.config import ConfigMixin


class DownShiftConv(nn.Conv2d):
    def forward(self, x: torch.Tensor):
        kh, kw = self.kernel_size
        x = F.pad(x, (kw // 2, kw // 2, kh - 1, 0))
        x = super().forward(x)
        return x


class DownRightShiftConv(nn.Conv2d):
    def forward(self, x: torch.Tensor):
        kh, kw = self.kernel_size
        x = F.pad(x, (kw - 1, 0, kh - 1, 0))
        x = super().forward(x)
        return x


class DownShiftConvTransposed(nn.ConvTranspose2d):
    def forward(self, x: torch.Tensor):
        x = super().forward(x)
        _, _, oh, ow = x.size()
        kh, kw = self.kernel_size
        sh, sw = self.stride
        x = x[..., : oh - kh + sh, kw // 2 : ow]
        return x


class DownRightShiftConvTransposed(nn.ConvTranspose2d):
    def forward(self, x: torch.Tensor):
        x = super().forward(x)
        _, _, oh, ow = x.size()
        kh, kw = self.kernel_size
        sh, sw = self.stride
        x = x[..., : oh - kh + sh, : ow - kw + sw]
        return x


def down_shift(x: torch.Tensor) -> torch.Tensor:
    return F.pad(x, (0, 0, 1, 0))[:, :, :-1, :]


def right_shift(x: torch.Tensor) -> torch.Tensor:
    return F.pad(x, (1, 0))[:, :, :, :-1]


def concat_elu(x: torch.Tensor) -> torch.Tensor:
    x = F.elu(torch.cat([x, -x], dim=1))
    return x


class GatedResidualDown(nn.Module):
    def __init__(self, channels, droprate: float = 0.0):
        super().__init__()

        self.vi_conv = DownShiftConv(channels * 2, channels, (2, 3))
        self.vo_conv = DownShiftConv(channels * 2, channels * 2, (2, 3))

        self.hi_conv = DownRightShiftConv(channels * 2, channels, (2, 2))
        self.v2h_conv = nn.Conv2d(channels * 2, channels, 1)
        self.ho_conv = DownRightShiftConv(channels * 2, channels * 2, (2, 2))

        self.dropout = nn.Dropout(droprate) if droprate > 0 else nn.Identity()

    def forward(self, xs: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        vx, hx = xs

        v_skip, h_skip = vx, hx

        vx = concat_elu(vx)
        vx = self.vi_conv(vx)
        vx = concat_elu(vx)
        vx = self.dropout(vx)
        vx = self.vo_conv(vx)
        va, vb = vx.chunk(2, dim=1)
        vx = v_skip + va * torch.sigmoid(vb)

        hx = concat_elu(hx)
        hx = self.hi_conv(hx)
        v2h = self.v2h_conv(concat_elu(vx))
        hx = hx + v2h
        hx = concat_elu(hx)
        hx = self.dropout(hx)
        hx = self.ho_conv(hx)
        ha, hb = hx.chunk(2, dim=1)
        hx = h_skip + ha * torch.sigmoid(hb)

        return (vx, hx)


class GatedResidualUp(nn.Module):
    def __init__(self, channels, droprate: float = 0.0):
        super().__init__()

        self.vi_conv = DownShiftConv(channels * 2, channels, (2, 3))
        self.v2v_conv = nn.Conv2d(channels * 2, channels, 1)
        self.vo_conv = DownShiftConv(channels * 2, channels * 2, (2, 3))

        self.hi_conv = DownRightShiftConv(channels * 2, channels, (2, 2))
        self.v2h_conv = nn.Conv2d(channels * 4, channels, 1)
        self.ho_conv = DownRightShiftConv(channels * 2, channels * 2, (2, 2))

        self.dropout = nn.Dropout(droprate) if droprate > 0 else nn.Identity()

    def forward(
        self, xs: tuple[torch.Tensor, torch.Tensor], d_xs: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vx, hx = xs
        d_vx, d_hx = d_xs

        v_skip, h_skip = vx, hx

        vx = concat_elu(vx)
        vx = self.vi_conv(vx)
        v2v = self.v2v_conv(concat_elu(d_vx))
        vx = vx + v2v
        vx = concat_elu(vx)
        vx = self.dropout(vx)
        vx = self.vo_conv(vx)
        va, vb = vx.chunk(2, dim=1)
        vx = v_skip + va * torch.sigmoid(vb)

        hx = concat_elu(hx)
        hx = self.hi_conv(hx)
        v2h = self.v2h_conv(concat_elu(torch.cat([vx, d_hx], dim=1)))
        hx = hx + v2h
        hx = concat_elu(hx)
        hx = self.dropout(hx)
        hx = self.ho_conv(hx)
        ha, hb = hx.chunk(2, dim=1)
        hx = h_skip + ha * torch.sigmoid(hb)

        return (vx, hx)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.v_down = DownShiftConv(channels, channels, (2, 3), (2, 2))
        self.h_down = DownRightShiftConv(channels, channels, (2, 2), (2, 2))

    def forward(self, xs: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        vx, hx = xs
        vx = self.v_down(vx)
        hx = self.h_down(hx)
        return (vx, hx)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.v_up = DownShiftConvTransposed(channels, channels, (2, 3), (2, 2))
        self.h_up = DownRightShiftConvTransposed(channels, channels, (2, 2), (2, 2))

    def forward(self, xs: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        vx, hx = xs
        vx = self.v_up(vx)
        hx = self.h_up(hx)
        return (vx, hx)


class PixelCNNpp(nn.Module, ConfigMixin):
    def __init__(
        self,
        num_layers_res: int = 5,
        channels: int = 64,
        num_logistic_mix: int = 10,
        image_channels: int = 1,
        droprate: float = 0.5,
    ):
        super().__init__()

        self.input_v = DownShiftConv(image_channels + 1, channels, (2, 3))
        self.input_h1 = DownShiftConv(image_channels + 1, channels, (1, 3))
        self.input_h2 = DownRightShiftConv(image_channels + 1, channels, (2, 1))

        layers = []
        for i in range(3):
            if i != 0:
                layers.append(Downsample(channels))
            for _ in range(num_layers_res):
                layers.append(GatedResidualDown(channels, droprate))
        self.downs = nn.ModuleList(layers)

        self.middle = GatedResidualDown(channels, droprate)

        layers = []
        for i in range(3):
            if i != 0:
                layers.append(Upsample(channels))
            for _ in range(num_layers_res):
                layers.append(GatedResidualUp(channels, droprate))
        self.ups = nn.ModuleList(layers)

        self.output_dims = (1 + 3 * image_channels) * num_logistic_mix
        self.output = nn.Conv2d(channels, self.output_dims, 1)

    def forward(self, x: torch.Tensor):
        x = F.pad(x, (0, 0, 0, 0, 0, 1), value=1)
        vx = down_shift(self.input_v(x))
        hx = down_shift(self.input_h1(x)) + right_shift(self.input_h2(x))

        xs = (vx, hx)

        down_feats = []
        for down in self.downs:
            xs = down(xs)
            if not isinstance(down, Downsample):
                down_feats.append(xs)

        xs = self.middle(xs)

        for up in self.ups:
            if isinstance(up, Upsample):
                xs = up(xs)
            else:
                xs = up(xs, down_feats.pop())

        output = self.output(xs[-1])
        return output
