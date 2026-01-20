import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_learning.IEDConformer.IED_deep_dearner import IEDDeepNet


class SeparableConv2d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=(1, 16), stride=(1, 1), padding=(0, 0), bias=False):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding,
            groups=in_ch, bias=bias
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class IEDConformer(IEDDeepNet):

    def __init__(
        self,
        n_channels,
        ts_len,
        n_classes: int = 4,
        ts_len_interpolate_to: int = 250,
        k: int = 40,
        t_kernel: int = 30,
        sep_kernel: int = 16,
        pool_kernel: int = 60,
        pool_stride: int = 15,
        # Transformer hyperparams (paper gives the idea; not all exact numbers specified)
        n_heads: int = 4,
        n_layers: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        mlp_hidden: int = 128,
    ):
        super().__init__(n_channels, ts_len, n_classes=n_classes)

        self.ts_len_interpolate_to = ts_len_interpolate_to

        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=k,
            kernel_size=(1, t_kernel),
            stride=(1, 1),
            padding=(0, t_kernel // 2),  # "same-ish" along time
            bias=False
        )

        self.spatial_conv = nn.Conv2d(
            in_channels=k,
            out_channels=k,
            kernel_size=(n_channels, 1),
            stride=(1, 1),
            padding=(0, 0),
            groups=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(k)
        self.act = nn.ELU()

        self.sep_conv = SeparableConv2d(
            in_ch=k,
            out_ch=k,
            kernel_size=(1, sep_kernel),
            stride=(1, 1),
            padding=(0, sep_kernel // 2),
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(k)

        self.avgpool = nn.AvgPool2d(
            kernel_size=(1, pool_kernel),
            stride=(1, pool_stride),
            padding=(0, 0)
        )
        self.drop = nn.Dropout(dropout)

        d_model = k
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # [B, S, E]
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.fc1 = nn.Linear(d_model, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, n_classes if n_classes != 2 else 1)

    def _core(self, x: torch.Tensor) -> torch.Tensor:
        if self.ts_len < self.ts_len_interpolate_to:
            x = F.interpolate(x, size=self.ts_len_interpolate_to, mode="linear", align_corners=True)

        x = x.unsqueeze(1)

        x = self.temporal_conv(x)
        x = self.spatial_conv(x)

        x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sep_conv(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.avgpool(x)
        x = self.drop(x)

        x = x.squeeze(2).transpose(1, 2)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.drop(x)
        logits = self.fc2(x)
        return logits

