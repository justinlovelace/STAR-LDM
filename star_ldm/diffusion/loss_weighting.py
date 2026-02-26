import torch
import torch.nn as nn
import numpy as np

# See https://arxiv.org/abs/2303.00848 for details on the weighting functions
def convert_epsweight_to_vweight(eps_weight, gamma):
    v_weight = torch.exp(torch.log(eps_weight)-torch.log(torch.exp(-gamma)+1))
    denom = v_weight.max()
    return torch.exp(torch.log(v_weight) - torch.log(denom))

def convert_epsweight_to_xweight(eps_weight, gamma, monotonic=False):
    x_weight = eps_weight*(torch.exp(gamma))
    if monotonic:
        monotonic_neg_x_weight = torch.cummax(x_weight, dim=0).values
        x_weight = monotonic_neg_x_weight
    denom = x_weight.max()
    return torch.exp(torch.log(x_weight) - torch.log(denom))

def get_loss_weighting(loss_weighting_name, **kwargs):
    assert loss_weighting_name == 'sigmoid', f"Only sigmoid loss weighting is supported, got: {loss_weighting_name}"
    return Sigmoid_Weighting(**kwargs)


class Sigmoid_Weighting(nn.Module):
    def __init__(self, gamma_shift, gamma_temp=1.0):
        super().__init__()
        self.min_gamma = -15
        self.max_gamma = 15
        self.gamma_shift = gamma_shift
        self.gamma_temp = gamma_temp

        gamma_linspace = torch.linspace(self.min_gamma, self.max_gamma, 10000)
        self.register_buffer('gamma_linspace', gamma_linspace)

        eps_weights = torch.sigmoid((self.gamma_shift-gamma_linspace)/self.gamma_temp)
        eps_weights = eps_weights/eps_weights.max()
        self.register_buffer('eps_weights', eps_weights)

        v_weights = convert_epsweight_to_vweight(self.eps_weights, gamma_linspace)
        self.register_buffer('v_weights', v_weights)

        x_weights = convert_epsweight_to_xweight(self.eps_weights, gamma_linspace)
        self.register_buffer('x_weights', x_weights)

    def v_loss_weighting(self, gamma):
        gamma_idx = torch.bucketize(gamma, self.gamma_linspace)
        return self.v_weights[gamma_idx]

    def eps_loss_weighting(self, gamma):
        gamma_idx = torch.bucketize(gamma, self.gamma_linspace)
        return self.eps_weights[gamma_idx]

    def x_loss_weighting(self, gamma):
        gamma_idx = torch.bucketize(gamma, self.gamma_linspace)
        return self.x_weights[gamma_idx]
