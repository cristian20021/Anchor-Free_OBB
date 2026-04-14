
import torch
import torch.nn as nn
import math

class HeadTower(nn.Module):
    def __init__(self, in_ch=256, num_convs=4):
        super().__init__()
        layers = []
        for _ in range(num_convs):
            # TODO — one block: Conv3×3 (same padding), GroupNorm(32), ReLU.
            # Why GroupNorm and not BatchNorm?
            # → at DOTA patch sizes, effective batch per GPU is tiny; BN stats
            #   become noisy. GN is independent of batch size.
            layers += [
                nn.Conv2d(in_ch, in_ch, 3, padding=1),
                nn.GroupNorm(32, 256),          # GroupNorm — how many groups? what size?
                nn.ReLU(inplace=True)
            ]
        self.tower = nn.Sequential(*layers)

    def forward(self, x):
        return self.tower(x)


class OBBHead(nn.Module):
    STRIDES = {0: 8, 1: 16, 2: 32, 3: 64}   # P3→P6

    def __init__(self, num_classes=15):
        super().__init__()
        self.tower = HeadTower()

        # TODO — three output branches (1×1 convs from 256ch).
        # cls:    256 → num_classes  (sigmoid at inference)
        # ctr:    256 → 1            (centerness, sigmoid)
        # reg:    256 → 5            (l, t, r, b, θ)
        self.cls_head = nn.Conv2d(256,num_classes,kernel_size=1)
        self.ctr_head = nn.Conv2d(256,1,kernel_size=1)
        self.reg_head = nn.Conv2d(256,5,kernel_size=1)

        # TODO — one trainable log-scale scalar per FPN level.
        # Initialise to 0 (exp(0)=1, i.e. no scaling at init).
        # Hint: nn.Parameter(torch.zeros(1)) × 4
        self.scales = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(4)])

    def forward(self, features):
        # features: list of [P3, P4, P5, P6] tensors
        out_cls, out_ctr, out_reg = [], [], []
        for i, feat in enumerate(features):
            x = self.tower(feat)
            out_cls.append(self.cls_head(x))
            out_ctr.append(self.ctr_head(x))

            # TODO — apply exp-scale to the first 4 dims (l,t,r,b),
            # leave θ (dim index 4) unscaled.
            raw = self.reg_head(x)
            ltrb = torch.exp(self.scales[i]) * raw[:, :4].exp()   # exp(scale) * raw[:, :4]
            theta = raw[:, 4:5]
            theta = theta.clamp(-math.pi / 2, 0)# raw[:, 4:5], clamped to (-π/2, 0]
            out_reg.append(torch.cat([ltrb, theta], dim=1))

        return out_cls, out_ctr, out_reg