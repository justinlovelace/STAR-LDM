import torch
import math
import numpy as np
from functools import partial


# Avoid log(0)
def log(t, eps = 1e-12):
    return torch.log(t.clamp(min = eps))

# noise schedule

def cosine_schedule(t, start = 0, end = 1, tau = 1, clip_min = 1e-9):
    power = 2 * tau
    output = torch.cos((t * (end - start) + start) * math.pi / 2) ** power
    return output.clamp(min = clip_min)

# converting gamma to alpha, sigma or logsnr
@torch.amp.autocast('cuda',enabled=False)
def log_snr_to_alpha2(log_snr):
    alpha2 = torch.sigmoid(log_snr)
    return alpha2

# Log-SNR shifting (https://arxiv.org/abs/2301.10972)
@torch.amp.autocast('cuda',enabled=False)
def alpha2_to_shifted_log_snr(alpha2, scale = 1):
    return (log(alpha2) - log(1 - alpha2)).clamp(min=-20, max=20) + 2*np.log(scale).item()

@torch.amp.autocast('cuda',enabled=False)
def time_to_alpha2(t, alpha2_schedule, scale):
    alpha2 = alpha2_schedule(t)
    shifted_log_snr = alpha2_to_shifted_log_snr(alpha2, scale = scale)
    return log_snr_to_alpha2(shifted_log_snr)

def get_scaled_noise_schedule(name, scale):
    assert name == 'cosine', f"Only cosine noise schedule is supported, got: {name}"
    return partial(time_to_alpha2, alpha2_schedule=cosine_schedule, scale=scale)
