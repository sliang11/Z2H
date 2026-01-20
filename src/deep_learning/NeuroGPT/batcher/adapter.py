# Module added by Shen Liang and Cihang Yu

import torch.nn as nn
import torch.nn.functional as F


class EEGSpatialAdapter(nn.Module):
    def __init__(self, input_channels, target_channels, input_length=None, target_length=500):
        """
        input_channels : nombre de canaux en entrée (C)
        target_channels : nombre de canaux en sortie
        input_length : nombre de points d'entrée
        target_length : nombre de points attendus par le modèle en sortie
        """
        super().__init__()
        self.input_channels = input_channels
        self.target_channels = target_channels
        self.input_length = input_length
        self.target_length = target_length

        self.spatial_conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=target_channels,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):
        # x: (batch, input_channels, L) où L = nombre de points par canal
        #exec(gen_cmd_print_variables('self.target_length, x.shape', pfx_msg='@@@@@@@@@@@@@ '))
        if (self.target_length is not None) and (x.shape[-1] != self.target_length):
            # Interpolation linéaire sur la dernière dimension (le temps)
            # torch.nn.functional.interpolate attend des tensors 3D (N, C, L)
            #exec(gen_cmd_print_variables("x.shape", pfx_msg="******* Before interpolation: "))
            x = F.interpolate(
                x,
                size=self.target_length,
                mode="linear",
                align_corners=True
            )
            #exec(gen_cmd_print_variables("x.shape", pfx_msg="******* After interpolation: "))
        # Passe la spatial conv (mapping de canaux)
        x = self.spatial_conv(x)  # (batch, target_channels, target_length)
        return x
