from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import ensure_tuple_rep


class TwoConv(nn.Sequential):

    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: str | tuple,
        norm: str | tuple,
        bias: bool,
        dropout: float | tuple = 0.0,
    ):

        super().__init__()

        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(
            spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


class Down(nn.Sequential):

    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: str | tuple,
        norm: str | tuple,
        bias: bool,
        dropout: float | tuple = 0.0,
    ):

        super().__init__()
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)


class UpCat(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: str | tuple,
        norm: str | tuple,
        bias: bool,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
        pre_conv: nn.Module | str | None = "default",
        interp_mode: str = "linear",
        align_corners: bool | None = True,
        halves: bool = True,
        is_pad: bool = True,
    ):

        super().__init__()
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)
        self.is_pad = is_pad

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):

        x_0 = self.upsample(x)

        if x_e is not None and torch.jit.isinstance(x_e, torch.Tensor):
            if self.is_pad:
                # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
                dimensions = len(x.shape) - 2
                sp = [0] * (dimensions * 2)
                for i in range(dimensions):
                    if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                        sp[i * 2 + 1] = 1
                x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0)

        return x


class BasicUNet(nn.Module):

    def __init__(
        self,
        spatial_dims_down: int = 2,
        spatial_dims_up: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features_down: Sequence[int] = (64, 64, 128, 256, 512),
        features_up: Sequence[int] = (16, 16, 32, 64, 128, 16),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
    ):

        super().__init__()

        fea_down = ensure_tuple_rep(features_down, 5)
        print(f"BasicUNet features: {fea_down}.")

        fea_up = ensure_tuple_rep(features_up, 6)
        print(f"BasicUNet features: {fea_up}.")

        self.conv_0 = TwoConv(spatial_dims_down, in_channels, fea_down[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims_down, fea_down[0], fea_down[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims_down, fea_down[1], fea_down[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims_down, fea_down[2], fea_down[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims_down, fea_down[3], fea_down[4], act, norm, bias, dropout)

        self.conv_convert_0 = Conv["conv", spatial_dims_up](1, fea_up[0], kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.conv_convert_1 = Conv["conv", spatial_dims_up](1, fea_up[1], kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.conv_convert_2 = Conv["conv", spatial_dims_up](1, fea_up[2], kernel_size=(8, 1, 1), stride=(8, 1, 1))
        self.conv_convert_3 = Conv["conv", spatial_dims_up](1, fea_up[3], kernel_size=(32, 1, 1), stride=(32, 1, 1))
        self.conv_convert_4 = Conv["conv", spatial_dims_up](1, fea_up[4], kernel_size=(128, 1, 1), stride=(128, 1, 1))

        self.upcat_4 = UpCat(spatial_dims_up, fea_up[4], fea_up[3], fea_up[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims_up, fea_up[3], fea_up[2], fea_up[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims_up, fea_up[2], fea_up[1], fea_up[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims_up, fea_up[1], fea_up[0], fea_up[5], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", spatial_dims_up](fea_up[5], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):

        x0 = self.conv_0(x)

        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        c0 = self.conv_convert_0(x0.unsqueeze(1))
        c1 = self.conv_convert_1(x1.unsqueeze(1))
        c2 = self.conv_convert_2(x2.unsqueeze(1))
        c3 = self.conv_convert_3(x3.unsqueeze(1))
        c4 = self.conv_convert_4(x4.unsqueeze(1))

        u4 = self.upcat_4(c4, c3)
        u3 = self.upcat_3(u4, c2)
        u2 = self.upcat_2(u3, c1)
        u1 = self.upcat_1(u2, c0)

        logits = self.final_conv(u1)
        return u1, u2, u3, u4, logits