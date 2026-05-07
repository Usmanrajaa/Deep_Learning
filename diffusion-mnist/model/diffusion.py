import torch
import torch.nn as nn
import numpy as np

class DDPM:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register('betas', betas)
        self.register('alphas', alphas)
        self.register('alphas_cumprod', alphas_cumprod)
        self.register('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def register(self, name, tensor):
        self.__dict__[name] = tensor

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise, noise

    @torch.no_grad()
    def p_sample(self, model, x, t):
        beta = self.betas[t].view(-1, 1, 1, 1)
        alpha = self.alphas[t].view(-1, 1, 1, 1)
        alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        eps_pred = model(x, t)
        mean = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_bar)) * eps_pred)
        if t[0] > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(beta)
        else:
            noise = 0.0
            sigma = 0.0
        return mean + sigma * noise

    @torch.no_grad()
    def sample(self, model, image_size, batch_size, channels, device):
        x = torch.randn(batch_size, channels, image_size, image_size, device=device)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch)
        x = (x - x.min()) / (x.max() - x.min())
        return x
