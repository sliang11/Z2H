import sys
sys.path.append('/linkhome/rech/genlpd01/ujv85fd/test/2-eeg-shen/')

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.resnet import Res1d18   # 假设你的文件叫 resnet.py，类名就是 Res1d18



# ========== Encoder ==========
class Res1d18Encoder(nn.Module):
    """只保留 Res1d18 的特征提取部分，输出 latent"""
    def __init__(self, base_model: Res1d18):
        super().__init__()
        self.feature = base_model._Res1d18__feature
        self.cnn = base_model._Res1d18__cnn

    def forward(self, x):
        x = self.feature(x)
        x = self.cnn(x)
        return x  # [B, 512, L_out]



# ========== Decoder ==========
class Res1d18Decoder(nn.Module):
    def __init__(self, dim_series, out_channels):
        super().__init__()
        self.dim_series = dim_series  # 保存原始输入长度

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose1d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.deconv5 = nn.ConvTranspose1d(64, out_channels, kernel_size=7, stride=2, padding=3, output_padding=1)

    def forward(self, z):
        out = self.deconv1(z)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.deconv5(out)
        # 修正：如果长度和原始不一致，强制插值/截断
        if out.shape[-1] != self.dim_series:
            out = F.interpolate(out, size=self.dim_series, mode="nearest")
        return out


# ========== Autoencoder ==========
class Res1d18Autoencoder(nn.Module):
    def __init__(self, base_model: Res1d18, in_channels: int, dim_series: int):
        super().__init__()
        self.encoder = Res1d18Encoder(base_model)
        self.decoder = Res1d18Decoder(dim_series, in_channels)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


if __name__ == "__main__":
    # 伪造一个 conf，至少要有这几个参数
    from util.conf import Conf
    conf = Conf()
    conf.setHP('num_input_channels', 5)   # 这里配置输入通道数
    conf.setHP('dim_series', 768)
    conf.setHP('num_class', 2)
    conf.setHP('threshold', 0.5)


    in_channels = conf.getHP('num_input_channels')
    base_model = Res1d18(conf)
    ae = Res1d18Autoencoder(base_model, in_channels, dim_series=conf.getHP('dim_series'))

    # 测试数据的通道数也要匹配 in_channels
    x = torch.randn(70, in_channels, conf.getHP('dim_series'))
    x_hat, z = ae(x)

    print("input :", x.shape)
    print("latent:", z.shape)
    print("recon :", x_hat.shape)
