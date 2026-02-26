import numpy as np
import torch
from torch import nn

def torch_sech(x):
    return 1/torch.cosh(x)

class CosineSampler(nn.Module):
    def __init__(self, gamma_min=-15, gamma_max=15, shift=0, device='cpu'):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.shift = shift

        t_min = self._logsnr_to_t(self.gamma_max - 2*self.shift, device=device)
        t_max = self._logsnr_to_t(self.gamma_min - 2*self.shift, device=device)

        self.register_buffer("t_min", t_min)
        self.register_buffer("t_max", t_max)

    def _logsnr_to_t(self, logsnr, device='cpu'):
        if not torch.is_tensor(logsnr):
            logsnr = torch.tensor(logsnr).to(device)
        t = (2/np.pi) * torch.atan(torch.exp(-logsnr/2 - self.shift/2))
        return t

    def _t_to_logsnr(self, t, t_min, t_max):
        adjusted_t = t_min + (t_max - t_min) * t
        logsnr = -2 * torch.log(torch.tan(adjusted_t * np.pi/2)) + self.shift
        return logsnr

    def _logsnr_to_density(self, logsnr, t_min, t_max):
        density = torch_sech(logsnr/2 - self.shift/2) / (2 * np.pi)
        density = density / (t_max - t_min)  # Truncation
        return density

    def sample(self, batch_size, device='cpu'):
        t = torch.rand((batch_size), device=device)
        logsnr = self._t_to_logsnr(t, self.t_min, self.t_max)
        density = self._logsnr_to_density(logsnr, self.t_min, self.t_max)
        return logsnr, density

class LossEMASampler(nn.Module):
    def __init__(self, n_bins=100, ema_decay=0.99, gamma_min=-15, gamma_max=15, train_schedule='cosine', cosine_shift=0):
        super().__init__()

        self.n_bins = n_bins
        self.ema_decay = ema_decay
        self.register_buffer("_loss_bins", torch.ones((n_bins), dtype=torch.float32))
        self.register_buffer("_unweighted_loss_bins", torch.ones((n_bins), dtype=torch.float32))

        self.register_buffer("_short_loss_bins", torch.ones((n_bins), dtype=torch.float32))
        self.register_buffer("_short_unweighted_loss_bins", torch.ones((n_bins), dtype=torch.float32))
        self.short_ema_decay = 0.5

        gamma_range = gamma_max - gamma_min
        self.bin_length = gamma_range/n_bins
        self.register_buffer("step", torch.tensor(0))
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

        assert train_schedule == 'cosine', f"Only cosine train schedule is supported, got: {train_schedule}"
        self.train_schedule = train_schedule
        self.sampler = CosineSampler(gamma_min=gamma_min, gamma_max=gamma_max, shift=cosine_shift)

    def weights(self, ):
        weights = self._loss_bins.clone()
        return weights

    def get_weighted_loss(self):
        return self._short_loss_bins.mean().item()

    @torch.no_grad()
    def update_ema_buffers(self, gamma, losses, unweighted_losses, fabric=None):
        bin_indices = ((gamma - self.gamma_min) / self.bin_length).long().clamp(0, self.n_bins - 1).to(losses.device)

        new_losses = torch.zeros_like(self._loss_bins)
        new_unweighted_losses = torch.zeros_like(self._unweighted_loss_bins)
        counts = torch.zeros_like(self._loss_bins)

        new_losses.scatter_add_(0, bin_indices, losses)
        new_unweighted_losses.scatter_add_(0, bin_indices, unweighted_losses)
        counts.scatter_add_(0, bin_indices, torch.ones_like(losses))

        mask = counts > 0
        new_losses[mask] /= counts[mask]
        new_unweighted_losses[mask] /= counts[mask]

        self._loss_bins[mask] = self.ema_decay * self._loss_bins[mask] + (1 - self.ema_decay) * new_losses[mask]
        self._short_loss_bins[mask] = self.short_ema_decay * self._short_loss_bins[mask] + (1 - self.short_ema_decay) * new_losses[mask]
        self._unweighted_loss_bins[mask] = self.ema_decay * self._unweighted_loss_bins[mask] + (1 - self.ema_decay) * new_unweighted_losses[mask]
        self._short_unweighted_loss_bins[mask] = self.short_ema_decay * self._short_unweighted_loss_bins[mask] + (1 - self.short_ema_decay) * new_unweighted_losses[mask]

        self.step += 1

    def sample(self, batch_size, device):
        gamma, density = self.sampler.sample(batch_size=batch_size, device=device)
        return gamma, density

    def get_loss_emas(self):
        loss_emas = {}
        for i in range(self.n_bins):
            gamma0 = i*self.bin_length + self.gamma_min
            gamma1 = (i+1)*self.bin_length + self.gamma_min
            loss_emas[(gamma0,gamma1)] = self._loss_bins[i].item()
        return loss_emas

    def get_unweighted_loss_emas(self):
        loss_emas = {}
        for i in range(self.n_bins):
            gamma0 = i*self.bin_length + self.gamma_min
            gamma1 = (i+1)*self.bin_length + self.gamma_min
            loss_emas[(gamma0,gamma1)] = self._unweighted_loss_bins[i].item()
        return loss_emas

    def get_normalized_loss_emas(self):
        loss_emas = {}
        denominator = self._loss_bins.sum().item()
        for i in range(self.n_bins):
            gamma0 = i*self.bin_length + self.gamma_min
            gamma1 = (i+1)*self.bin_length + self.gamma_min
            loss_emas[(gamma0,gamma1)] = self._loss_bins[i].item() / denominator
        return loss_emas

    def get_cdf(self):
        cdf_dict = {}
        weights = self.weights()
        cdf = torch.cumsum(weights/weights.sum(), dim=0)
        for i in range(self.n_bins):
            gamma0 = i*self.bin_length + self.gamma_min
            gamma1 = (i+1)*self.bin_length + self.gamma_min
            cdf_dict[(gamma0,gamma1)] = cdf[i].item()
        return cdf_dict
