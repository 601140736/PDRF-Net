# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "ADD",
    "RepConv",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)

# #定义基于CLIP和mamba的融合模块
# from .mamba_base.cross_mamba_simple import Mamba
#
# class SingleMambaBlock(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         # self.norm1 = nn.LayerNorm(dim)
#         self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v6',
#                            if_devide_out=True, use_norm=True)
#
#     def forward(self, input):
#         # input: (B, N, C)
#         skip = input
#         input = self.norm(input)
#         output = self.block(input)
#         # output = self.norm1(output)
#         return output + skip


# class CrossMambaBlock(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.norm0 = nn.LayerNorm(dim)
#         self.norm1 = nn.LayerNorm(dim)
#         # self.norm2 = nn.LayerNorm(dim)
#         self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v7',
#                            if_devide_out=True, use_norm=True)
#
#     def forward(self, input0, input1):
#         # input0: (B, N, C) | input1: (B, N, C)
#         skip = input0
#         input0 = self.norm0(input0)
#         input1 = self.norm1(input1)
#         output = self.block(input0, extra_emb=input1)
#         # output = self.norm2(output)
#         return output + skip

#
# class FusionMamba(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.cross = CrossMambaBlock(dim)
#
#     def forward(self, vi, ir):
#         b, c, h, w = vi.shape
#         vi = rearrange(vi, 'b c h w -> b (h w) c', h=h, w=w)
#         ir = rearrange(ir, 'b c h w -> b (h w) c', h=h, w=w)
#         fusion = self.cross(ir, vi)
#         fusion = rearrange(fusion, 'b (h w) c -> b c h w', h=h, w=w)
#         return fusion
#
# class FusionMamba_clip(nn.Module):
#     def __init__(self, dim):
#         super(FusionMamba_clip, self).__init__()
#         self.cross_mamba = CrossMambaBlock(dim)
#         self.out_proj = nn.Linear(dim, dim)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, feature1, feature2):
#         '''
#         feature1 为image 的向量，为b,c,1,1,
#         feature2 为clip输出的场景向量，为b,512->b,c
#         '''
#         feature1 = rearrange(feature1, 'b c 1 1 -> b 1 c')
#         feature2 = rearrange(feature2, 'b c -> b 1 c')
#
#         fusion = self.cross_mamba(feature1, feature2)
#         fusion = self.out_proj(fusion)
#         output = rearrange(fusion, 'b 1 c -> b c 1 1')
#         return self.sigmoid(output)



# class ImageProjectionHead(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(ImageProjectionHead, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, 1024),
#             nn.LeakyReLU(),
#             nn.Linear(1024, output_dim)
#         )
#
#     def forward(self, x):
#         x = self.fc(x)
#         return x
#
# # 添加一个线性层来调整维度
# from torchvision import datasets, transforms, models
# # 使用预训练的 ResNet 作为图像编码器
# resnet_model = models.resnet50(pretrained=True)  # 可以选择 resnet18, resnet34, resnet50, resnet101 等
# # 去掉最后的全连接层
# resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])  # 仅保留卷积部分
# class ResNetWithProjection(nn.Module):
#     def __init__(self, base_model=resnet_model, output_dim=512):
#         super(ResNetWithProjection, self).__init__()
#         self.base_model = base_model
#         self.fc = nn.Linear(2048, output_dim)  # 2048是ResNet的特征维度
#
#     def forward(self, x):
#         x = self.base_model(x)
#         x = x.view(x.size(0), -1)  # Flatten the output
#         x = self.fc(x)  # 调整维度
#         return x

# model_clip_cls = ResNetWithProjection()
# model_clip_cls.load_state_dict(torch.load('runs/resnet_clip.pth'))
# model_clip_cls.eval()


# class CLIP_deal(nn.Module):
#     def __init__(self, model=model_clip_cls):
#         super(CLIP_deal, self).__init__()
#         # 使用预训练的 ResNet 作为图像编码器
#         self.model = model
#         self.model.eval()
#
#     def forward(self, x):
#         with torch.no_grad():
#             return self.model(x)



# class Deal(nn.Module):
#     def __init__(self):
#         super(Deal, self).__init__()
#
#     def forward(self, x):
#         return 0*x


from einops import rearrange
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, model_dim, num_heads=1):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        assert (self.head_dim * num_heads == model_dim), "model_dim must be divisible by num_heads"

        self.query_vis = nn.Linear(model_dim, model_dim)
        self.key_vis = nn.Linear(model_dim, model_dim)
        self.value_vis = nn.Linear(model_dim, model_dim)

        self.query_inf = nn.Linear(model_dim, model_dim)
        self.key_inf = nn.Linear(model_dim, model_dim)
        self.value_inf = nn.Linear(model_dim, model_dim)

        self.fc_out_vis = nn.Linear(model_dim, model_dim)
        self.fc_out_inf = nn.Linear(model_dim, model_dim)

    def forward(self, vis, inf):
        b, c, h, w = vis.shape
        vis = rearrange(vis, 'b c h w -> b (h w) c', h=h, w=w)
        inf = rearrange(inf, 'b c h w -> b (h w) c', h=h, w=w)
        batch_size, seq_length, model_dim = vis.shape

        # vis -> Q, K, V
        Q_vis = self.query_vis(vis)
        K_vis = self.key_vis(vis)
        V_vis = self.value_vis(vis)

        # inf -> Q, K, V
        Q_inf = self.query_inf(inf)
        K_inf = self.key_inf(inf)
        V_inf = self.value_inf(inf)

        # Reshape for multi-head attention
        Q_vis = Q_vis.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,
                                                                                            2)  # B, N, C --> B, n_h, N, d_h
        K_vis = K_vis.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V_vis = V_vis.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        Q_inf = Q_inf.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K_inf = K_inf.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V_inf = V_inf.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Cross attention: vis Q with inf K and inf Q with vis K
        # Q_vis 的形状为 (batch_size, num_heads, seq_length, head_dim)
        # K_inf 的形状为 (batch_size, num_heads, head_dim, seq_length)
        # 矩阵乘法后，scores_vis_inf 的形状为 (batch_size, num_heads, seq_length, seq_length)
        scores_vis_inf = torch.matmul(Q_vis, K_inf.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))
        scores_inf_vis = torch.matmul(Q_inf, K_vis.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))

        attention_inf = torch.softmax(scores_vis_inf, dim=-1)
        attention_vis = torch.softmax(scores_inf_vis, dim=-1)

        # attention_vis_inf 的形状为 (batch_size, num_heads, seq_length, seq_length)
        # V_inf 的形状为 (batch_size, num_heads, seq_length, head_dim)
        # out_vis_inf 的形状为 (batch_size, num_heads, seq_length, head_dim)
        out_inf = torch.matmul(attention_inf, V_inf)
        out_vis = torch.matmul(attention_vis, V_vis)

        # Concatenate and project back to the original dimension
        out_vis = out_vis.transpose(1, 2).contiguous().view(batch_size, seq_length, model_dim)
        out_inf = out_inf.transpose(1, 2).contiguous().view(batch_size, seq_length, model_dim)

        # out 的形状为 (batch_size, seq_length, model_dim)
        out_vis = self.fc_out_vis(out_vis)
        out_inf = self.fc_out_inf(out_inf)

        return rearrange(out_vis, 'b (h w) c -> b c h w', h=h, w=w), rearrange(out_inf, 'b (h w) c -> b c h w', h=h, w=w)

#-------------------------------------------#
# class enhance(nn.Module):
#     def __init__(self):
#         super(enhance, self).__init__()
#         # ----增强空间---
#         self.conv1 = nn.Conv2d(in_channels=4, out_channels=1, stride=1, kernel_size=7, padding=3)
#         self.sigmoid = nn.Sigmoid()
#         self.catt = MultiHeadCrossAttention(model_dim=2)
#
#     def forward(self,x):
#         vis = x[0]
#         inf = x[1]
#
#         ave1 = torch.mean(vis, dim=1, keepdim=True)
#         ave2, _ = torch.max(vis, dim=1, keepdim=True)
#         ave12 = torch.cat([ave1,ave2],dim=1)
#
#         ave3 = torch.mean(inf, dim=1, keepdim=True)
#         ave4,_ = torch.max(inf, dim=1, keepdim=True)
#         ave34 = torch.cat([ave3, ave4],dim=1)
#         ave12, ave34 = self.catt(ave12, ave34)
#         ave1234=self.sigmoid(self.conv1(torch.cat([ave12,ave34], dim=1)))
#         return ave1234*x[0], x[1]
#----------------------------------------
# class CMFusion(nn.Module):
#     def __init__(self, dim):
#         super(CMFusion, self).__init__()
#         self.enhance = enhance()
#         self.conv = nn.Conv2d(in_channels=dim*2, out_channels=dim, stride=1, padding=0, kernel_size=1)
#
#     def forward(self, x):
#         vi, ir = x[0],x[1]
#         vi,ir = self.enhance([vi,ir])
#         return self.conv(torch.cat([vi,ir],dim=1))


class ADD(nn.Module):
    def __init__(self, dim):
        super(ADD, self).__init__()
        self.d = dim

    def forward(self, x):
        return x[0] + x[1]
