import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from autoregressive.activation import get_activation_cls
from autoregressive.config import ConfigMixin


class CausalAttention(nn.Module):
    def __init__(
        self,
        dims: int,
        num_heads: int = 8,
        head_dims: int | None = None,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()

        self.dims = dims
        self.num_heads = num_heads
        self.head_dims = head_dims or self.dims // num_heads
        self.inner_dims = self.num_heads * self.head_dims
        self.scale = self.head_dims**-0.5

        self.to_qkv = nn.Linear(self.dims, self.inner_dims * 3, bias=qkv_bias)
        self.proj = nn.Linear(self.inner_dims, self.dims, bias=proj_bias)
        self.proj_dropout = nn.Dropout(proj_dropout) if proj_dropout > 0.0 else nn.Identity()

        self.attn_dropout = nn.Dropout(attn_dropout)

        self.use_flash = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = x.size()

        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.reshape(B, N, self.num_heads, self.head_dims).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dims).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dims).transpose(1, 2)

        if self.use_flash:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=True
            )
        else:
            torch._assert(attn_mask is not None, 'attn_mask required.')
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))
            attn = attn.softmax(-1)
            attn = self.attn_dropout(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, self.inner_dims)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, 1, dims))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()

        denom = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).rsqrt()
        x = x * denom

        return x.to(dtype) * self.scale


def Mlp(dims: int, ratio: float = 4.0, bias: bool = True, activation: str = 'gelu', dropout: float = 0.0) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dims, int(dims * ratio), bias=bias),
        nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        get_activation_cls(activation)(),
        nn.Linear(int(dims * ratio), dims, bias=bias),
        nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
    )


class Block(nn.Module):
    def __init__(
        self,
        dims: int,
        num_heads: int = 8,
        head_dims: int | None = None,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        bias: bool = True,
        activation: str = 'gelu',
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        dropout: float = 0.0,
        rms_norm: bool = True,
        eps: float = 1e-6,
        layer_scale_init: float | None = 1e-5,
    ):
        super().__init__()

        NormCls = RMSNorm if rms_norm else nn.LayerNorm
        self.norm_attn = NormCls(dims, eps=eps)
        self.attn = CausalAttention(
            dims,
            num_heads=num_heads,
            head_dims=head_dims,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )
        self.norm_mlp = NormCls(dims, eps=eps)
        self.mlp = Mlp(dims, mlp_ratio, bias=bias, activation=activation, dropout=dropout)

        if layer_scale_init is not None:
            self.attn_scale = nn.Parameter(torch.ones(1) * layer_scale_init)
            self.mlp_scale = nn.Parameter(torch.ones(1) * layer_scale_init)
        else:
            self.register_buffer('attn_scale', torch.ones(1))
            self.register_buffer('mlp_scale', torch.ones(1))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn_scale * self.attn(self.norm_attn(x), attn_mask=attn_mask)
        x = x + self.mlp_scale * self.mlp(self.norm_mlp(x))
        return x


class GPT2(nn.Module):
    def __init__(
        self,
        seq_length: int,
        dims: int,
        num_vocabs: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        head_dims: int | None = None,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        bias: bool = True,
        activation: str = 'gelu',
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        dropout: float = 0.0,
        rms_norm: bool = True,
        eps: float = 1e-6,
        layer_scale_init: float | None = 1e-5,
    ):
        super().__init__()

        # start of sequence
        self.sos = nn.Parameter(torch.randn(1, 1, dims))
        # position encoding
        self.pe = nn.Parameter(torch.randn(1, seq_length, 1))
        # causal mask
        self.register_buffer('attn_mask', torch.tril(torch.ones(1, 1, seq_length, seq_length)))

        # Embed color palette.
        self.embed = nn.Embedding(num_vocabs, dims)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                Block(
                    dims=dims,
                    num_heads=num_heads,
                    head_dims=head_dims,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    bias=bias,
                    activation=activation,
                    mlp_ratio=mlp_ratio,
                    attn_dropout=attn_dropout,
                    proj_dropout=proj_dropout,
                    dropout=dropout,
                    rms_norm=rms_norm,
                    eps=eps,
                    layer_scale_init=layer_scale_init,
                )
            )

        self.norm = RMSNorm(dims, eps=eps) if rms_norm else nn.LayerNorm(dims, eps=eps)
        self.head = nn.Linear(dims, num_vocabs, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            C = x.size(1)
            torch._assert(C == 1, 'If ndim==4, channels should be 1')
            x = x.squeeze()
        if x.ndim == 3:
            B, H, W = x.size()
            x = x.reshape(B, H * W)

        # embed
        hidden_state = self.embed(x)

        # prepend sos.
        sos = self.sos.repeat(hidden_state.size(0), 1, 1)
        hidden_state = torch.cat([sos, hidden_state[:, :-1]], dim=1)

        # add pe
        hidden_state = hidden_state + self.pe

        for layer in self.layers:
            hidden_state = layer(hidden_state, attn_mask=self.attn_mask)

        hidden_state = self.norm(hidden_state)
        logits = self.head(hidden_state)

        return logits


class ImageGPT(nn.Module, ConfigMixin):
    def __init__(
        self,
        color_palette: str,
        seq_length: int,
        dims: int,
        num_layers: int = 12,
        num_heads: int = 8,
        head_dims: int | None = None,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        bias: bool = True,
        activation: str = 'gelu',
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        dropout: float = 0.0,
        rms_norm: bool = True,
        eps: float = 1e-6,
        layer_scale_init: float | None = 1e-5,
    ):
        super().__init__()

        color_palette = np.load(color_palette)
        self.register_buffer('color_palette', torch.from_numpy(color_palette))
        self.num_colors = num_vocabs = self.color_palette.size(0)

        self.gpt = GPT2(
            seq_length=seq_length,
            dims=dims,
            num_vocabs=num_vocabs,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dims=head_dims,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            bias=bias,
            activation=activation,
            mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            dropout=dropout,
            rms_norm=rms_norm,
            eps=eps,
            layer_scale_init=layer_scale_init,
        )

    def quantize(self, image: torch.Tensor) -> torch.Tensor:
        B, _, H, W = image.size()
        image = image.permute(0, 2, 3, 1).contiguous()
        image_flattened = image.reshape(-1, self.color_palette.size(1))

        closest_indices = torch.argmin(torch.cdist(image_flattened, self.color_palette), dim=1)
        closest_indices = closest_indices.reshape(B, H, W)
        return closest_indices

    def forward(self, x: torch.Tensor, return_labels: bool = False) -> torch.Tensor:
        labels = self.quantize(x)
        logits = self.gpt(labels)  # [BLN]

        B, H, W = labels.size()
        logits = logits.transpose(-1, -2)  # [BNL]
        logits = logits.reshape(B, -1, H, W)  # [BNHW]

        if return_labels:
            return logits, labels

        return logits


@torch.no_grad()
def sample_image(model: ImageGPT, size: tuple[int, int, int, int], device: torch.device) -> torch.Tensor:
    model.eval()

    # DDP stuff.
    unwrapped_model: ImageGPT = model.module if hasattr(model, 'module') else model

    B, H, W = size
    color_palette = unwrapped_model.color_palette
    image = torch.zeros(B, color_palette.size(-1), H, W, device=device)

    for h in range(H):
        for w in range(W):
            logits = model(image)
            probs = logits[:, :, h, w].softmax(1)
            index = torch.multinomial(probs, 1).squeeze()
            image[:, :, h, w] = color_palette[index]

    model.train()
    return image
