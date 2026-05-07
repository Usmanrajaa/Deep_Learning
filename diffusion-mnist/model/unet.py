import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        emb = torch.exp(torch.arange(half, dtype=torch.float, device=t.device) * (
            -torch.log(torch.tensor(10000.0)) / (half - 1)
        ))
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        h = h + self.time_proj(temb).unsqueeze(-1).unsqueeze(-1)
        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)
        return h + self.shortcut(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, time_dim=128):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.down1 = ResBlock(base_ch, base_ch, time_dim)
        self.down2 = ResBlock(base_ch, base_ch*2, time_dim)
        self.down3 = ResBlock(base_ch*2, base_ch*4, time_dim)
        self.up1 = ResBlock(base_ch*4 + base_ch*2, base_ch*2, time_dim)
        self.up2 = ResBlock(base_ch*2 + base_ch, base_ch, time_dim)
        self.up3 = ResBlock(base_ch*2, base_ch, time_dim)
        self.conv_out = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def forward(self, x, t):
        temb = self.time_emb(t.float())
        temb = self.time_mlp(temb)
        x0 = self.conv_in(x)
        x1 = self.down1(x0, temb)
        x2 = self.down2(nn.functional.avg_pool2d(x1, 2), temb)
        x3 = self.down3(nn.functional.avg_pool2d(x2, 2), temb)
        x = nn.functional.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up1(torch.cat([x, x2], dim=1), temb)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up2(torch.cat([x, x1], dim=1), temb)
        x = self.up3(torch.cat([x, x0], dim=1), temb)
        return self.conv_out(x)
