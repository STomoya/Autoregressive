"""From: https://github.com/kamenbliznashki/pixel_models/blob/b8a574664827f1d305091d6767be946268d76491/pixelcnnpp.py"""

import torch
import torch.nn.functional as F

# --------------------
# Loss functions
# --------------------


def discretized_mix_logistic_loss(output: torch.Tensor, target: torch.Tensor, n_bits: int):
    """log likelihood for mixture of discretized logistics"""
    # shapes
    B, C, H, W = target.shape
    n_mix = output.shape[1] // (1 + 3 * C)  # logits x 1, mean x 3, log scale x 3, coeffs x 3

    # unpack params of mixture of logistics
    logits = output[:, :n_mix, :, :]  # (B, n_mix, H, W)
    output = output[:, n_mix:, :, :].reshape(B, 3 * n_mix, C, H, W)
    means, logscales, coeffs = output.split(n_mix, 1)  # (B, n_mix, C, H, W)
    logscales = logscales.clamp(min=-7)
    coeffs = coeffs.tanh()

    # adjust means of channels based on preceding subpixel (cf PixelCNN++ eq 3)
    target = target.unsqueeze(1).expand_as(means)
    if C != 1:
        m1 = means[:, :, 0, :, :]
        m2 = means[:, :, 1, :, :] + coeffs[:, :, 0, :, :] * target[:, :, 0, :, :]
        m3 = (
            means[:, :, 2, :, :]
            + coeffs[:, :, 1, :, :] * target[:, :, 0, :, :]
            + coeffs[:, :, 2, :, :] * target[:, :, 1, :, :]
        )
        means = torch.stack([m1, m2, m3], 2)  # out (B, n_mix, C, H, W)

    # log prob components
    scales = torch.exp(-logscales)
    plus = scales * (target - means + 1 / (2**n_bits - 1))
    minus = scales * (target - means - 1 / (2**n_bits - 1))

    # partition the logistic pdf and cdf for target in [<-0.999, mid, >0.999]
    # 1. target<-0.999 ie edge case of 0 before scaling
    cdf_minus = torch.sigmoid(minus)
    log_one_minus_cdf_minus = -F.softplus(minus)
    # 2. target>0.999 ie edge case of 255 before scaling
    cdf_plus = torch.sigmoid(plus)
    log_cdf_plus = plus - F.softplus(plus)
    # 3. target in [-.999, .999] is log(cdf_plus - cdf_minus)

    # compute log probs:
    # 1. for target < -0.999, return log_cdf_plus
    # 2. for target > 0.999,  return log_one_minus_cdf_minus
    # 3. target otherwise,    return cdf_plus - cdf_minus
    log_probs = torch.where(
        target < -0.999,
        log_cdf_plus,
        torch.where(target > 0.999, log_one_minus_cdf_minus, torch.log((cdf_plus - cdf_minus).clamp(min=1e-12))),
    )
    log_probs = log_probs.sum(2) + F.log_softmax(
        logits, 1
    )  # log_probs sum over channels (cf eq 3), softmax over n_mix components (cf eq 1)

    # marginalize over n_mix components and return negative log likelihood per data point
    return -log_probs.logsumexp(1).sum([1, 2])  # out (B,)


# --------------------
# Sampling and generation functions
# --------------------


def sample_from_discretized_mix_logistic(output, image_dims):
    # shapes
    B, _, H, W = output.shape
    C = image_dims[0]  # 3
    n_mix = output.shape[1] // (1 + 3 * C)

    # unpack params of mixture of logistics
    logits = output[:, :n_mix, :, :]
    output = output[:, n_mix:, :, :].reshape(B, 3 * n_mix, C, H, W)
    means, logscales, coeffs = output.split(n_mix, 1)  # each out (B, n_mix, C, H, W)
    logscales = logscales.clamp(min=-7)
    coeffs = coeffs.tanh()

    # sample mixture indicator
    argmax = torch.argmax(logits - torch.log(-torch.log(torch.empty_like(logits).uniform_(1e-5, 1 - 1e-5))), dim=1)
    sel = torch.eye(n_mix, device=logits.device)[argmax]
    sel = sel.permute(0, 3, 1, 2).unsqueeze(2)  # (B, n_mix, 1, H, W)

    # select mixture components
    means = means.mul(sel).sum(1)
    logscales = logscales.mul(sel).sum(1)
    coeffs = coeffs.mul(sel).sum(1)

    # sample from logistic using inverse transform sampling
    u = torch.rand_like(means).uniform_(1e-5, 1 - 1e-5)
    x = means + logscales.exp() * (torch.log(u) - torch.log1p(-u))  # logits = inverse logistic

    if C == 1:
        return x.clamp(-1, 1)
    else:
        x0 = torch.clamp(x[:, 0, :, :], -1, 1)
        x1 = torch.clamp(x[:, 1, :, :] + coeffs[:, 0, :, :] * x0, -1, 1)
        x2 = torch.clamp(x[:, 2, :, :] + coeffs[:, 1, :, :] * x0 + coeffs[:, 2, :, :] * x1, -1, 1)
        return torch.stack([x0, x1, x2], 1)  # out (B, C, H, W)


@torch.no_grad()
def sample_image(model: torch.nn.Module, size: tuple[int, int, int, int], device: torch.device) -> torch.Tensor:
    B, C, H, W = size

    out = torch.zeros(size, device=device)
    for y in range(H):
        for x in range(W):
            logits = model(out)
            out[:, :, y, x] = sample_from_discretized_mix_logistic(logits, (C, H, W))[:, :, y, x]
    return out
