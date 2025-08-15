import math

from einops.layers.torch import Rearrange


from models.vmamba import VSSBlock, Permute
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import cv2
# from sam2.modeling.memory_attention import MemoryAttentionLayer, MemoryAttention
from sam2.modeling.sam.prompt_encoder import PromptEncoder
import numpy as np
sys.path.append('./')


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class FD(nn.Module):
    def __init__(self, ):
        super(FD, self).__init__()

    def forward(self, x):
        feature_high, feature_low = self.fft(x)
        return feature_high, feature_low

    def shift(self, x):
        _, _, height, width = x.shape
        return torch.roll(x, shifts=(int(height / 2), int(width / 2)), dims=(2, 3))

    def unshift(self, x):
        _, _, height, width = x.shape
        return torch.roll(x, shifts=(-int(height / 2), -int(width / 2)), dims=(2, 3))

    def fft(self, x):
        masking = torch.zeros(x.shape).to(x.device)
        transformed = torch.fft.fft2(x, norm='forward', dim=(-2, -1))
        transformed = self.shift(transformed)
        high_freq = transformed * (1 - masking)
        high_detail = self.unshift(high_freq)
        high_detail = torch.fft.ifft2(
            high_detail, norm='forward', dim=(-2, -1))
        high_detail = torch.abs(high_detail)
        low_freq = transformed * masking
        low_detail = self.unshift(low_freq)
        low_detail = torch.fft.ifft2(low_detail, norm='forward', dim=(-2, -1))
        low_detail = torch.abs(low_detail)
        return high_detail, low_detail


class Layer(nn.Module):
    def __init__(self, sub_net, weight_sub, num_sub):
        super().__init__()
        assert len(sub_net) > 0
        self.nets = nn.ModuleList(sub_net)
        self.wet = weight_sub
        self.num_net = num_sub

    def forward(self, inputs, k):
        out = self.wet(inputs)
        weights = F.softmax(out, dim=1, dtype=torch.float).to(inputs.dtype)
        topk_weights, topk_nets = torch.topk(weights, self.num_net)
        out = inputs.clone()
        exp_weights = torch.zeros_like(weights)
        exp_weights.scatter_(1, topk_nets, weights.gather(1, topk_nets))
        for i, net in enumerate(self.nets):
            out += net(inputs, k) * exp_weights[:, i:i + 1, None, None]
        return out


class SubNet(nn.Module):
    def __init__(self, in_ch, low_dim):
        super().__init__()
        self.conv_1 = nn.Conv2d(
            in_channels=in_ch, out_channels=low_dim, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(
            in_channels=in_ch, out_channels=low_dim, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=low_dim,
                                out_channels=in_ch, kernel_size=1)

    def forward(self, x, k):
        x = self.conv_1(x)
        k = self.conv_2(k)
        x = k * x
        x = self.conv_3(x)
        return x


class WET(nn.Module):
    def __init__(self, in_ch, num_experts):
        super().__init__()

        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('b c 1 1 -> b c'),
            nn.Linear(in_ch, num_experts, bias=False),
        )

    def forward(self, x):
        return self.body(x)


class FEM(nn.Module):
    def __init__(self,
                 in_ch: int,
                 num_nets: int,
                 topk: int):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_ch, 2 * in_ch, kernel_size=3, padding=1)
        self.sub_layer = Layer(
            sub_net=[SubNet(in_ch=in_ch, low_dim=in_ch)
                     for i in range(num_nets)],
            weight_sub=WET(in_ch=in_ch, num_experts=num_nets),
            num_sub=topk,
        )

    def forward(self, x):
        x = self.conv_1(x)
        x, k = torch.chunk(x, chunks=2, dim=1)
        x = self.sub_layer(x, k)
        return x


class FD_FEM(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.spilit = FD()

        self.attn1 = FEM(in_ch, num_nets=3, topk=2)
        self.attn2 = FEM(in_ch, num_nets=3, topk=2)

    def forward(self, x):
        low, high = self.spilit(x)
        low = self.attn1(low)
        high = self.attn2(high)
        out = low + high
        return out + x


from mamba_ssm import Mamba
import torch.nn.functional as F
from einops import pack, repeat, unpack
class GM(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GM, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):
        embedding = (x.pow(2).sum((2, 3), keepdim=True) +self.epsilon).pow(0.5) * self.alpha
        norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        gate = 1. + torch.tanh(embedding * norm + self.beta)
        return x * gate


import torch.nn as nn
import math


class ECA_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        out = x * y.expand_as(x)
        return out

class RefineModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.ChannelAttention_CGA=ECA_block(dim)

        self.pa = PixelAttention_CGA(dim)
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        act = torch.cat([avg_out, max_out], dim=1)
        act = self.conv(act)
        act = self.sigmoid(act)
        # X2=self.ChannelAttention_CGA(x)
        # pattn2 = self.sigmoid(self.pa( act * x, X2))
        # result =pattn2 * x + (1 - pattn2) * x
        out = act * x
        return out


import torch.nn as nn
import torch
from einops import rearrange


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


class SpatialAttention_CGA(nn.Module):
    def __init__(self):
        super(SpatialAttention_CGA, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention_CGA(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention_CGA, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention_CGA(nn.Module):
    def __init__(self, dim):
        super(PixelAttention_CGA, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = rearrange(x2, 'b c t h w -> b (c t) h w')
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention_CGA()
        self.ca = ChannelAttention_CGA(dim, reduction)
        self.pa = PixelAttention_CGA(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, y = data
        initial =data
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * data + (1 - pattn2) * data
        result = self.conv(result)
        return result


class PartFusionModule(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.scale1 = nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False)
        self.scale2 = nn.Conv2d(in_ch, in_ch, 5, padding=2, bias=False)
        self.scale3 = nn.Conv2d(in_ch, in_ch, 7, padding=3, bias=False)

        self.ref1 =RefineModule(in_ch)
        self.ref2 =RefineModule(in_ch)
        self.ref3 =RefineModule(in_ch)

    def forward(self, x):
        x1 = self.scale1(x)
        x2 = self.scale2(x)
        x3 = self.scale3(x)
        x1 = self.ref1(x1)
        x2 = self.ref2(x2)
        x3 = self.ref3(x3)
        out = x1 + x2 + x3
        return out + x


class FusionModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(FusionModule, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.faffm = PartFusionModule(in_ch=out_channels)

    def forward(self, x):
        identity = x
        x = self.faffm(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class GambaCell(nn.Module):
    def __init__(self, n_dims, height, width):
        super().__init__()
        self.rel_h = nn.Parameter(torch.randn([1, n_dims, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, n_dims, width, 1]), requires_grad=True)
        self.mamba_block = Mamba(d_model=n_dims, d_state=64, d_conv=4, expand=2)

        self.gate = nn.Sequential(
            nn.Conv1d(n_dims, n_dims, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, H, W) -> Reshape to (B, seq_len, C)
        B, C, height, width = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # (B, seq_len, C)

        position = (self.rel_h + self.rel_w).view(1, C, -1).permute(0, 2, 1)  # (B, seq_len, C)
        x = x + position

        # Fix: Permute before Conv1d
        x_permuted = x.permute(0, 2, 1)  # (B, C, seq_len)
        context = self.gate(x_permuted)  # Apply Conv1d
        context = context.permute(0, 2, 1)  # Back to (B, seq_len, C)

        x_mamba = self.mamba_block(x)

        # Combine outputs
        x = x_mamba * context
        x = x.view(B, C, height, width)
        return x


class MHSA(nn.Module):  # with register tokens (iiANET)
    def __init__(self, n_dims, width, height, head, num_register_tokens=1):
        super(MHSA, self).__init__()
        self.head = head

        self.Q = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.K = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.V = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, head, n_dims // head, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, head, n_dims // head, width, 1]), requires_grad=True)

        self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, width * height, width * height))
        self.register_tokens_v = nn.Parameter(torch.randn(num_register_tokens, n_dims // head, width * height))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()

        q = self.Q(x).view(n_batch, self.head, C // self.head, -1)
        k = self.K(x).view(n_batch, self.head, C // self.head, -1)
        v = self.V(x).view(n_batch, self.head, C // self.head, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.head, C // self.head, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)
        energy = content_content + content_position

        r_qk = repeat(
            self.register_tokens,
            'n w h -> b n w h',
            b=n_batch
        )

        r_v = repeat(
            self.register_tokens_v,
            'n w h -> b n w h',
            b=n_batch
        )

        energy, _ = pack([energy, r_qk], 'b * w h')
        v, ps = pack([v, r_v], 'b * d h')

        attention = self.softmax(energy)
        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out, _ = unpack(out, ps, 'b * d h')
        out = out.view(n_batch, C, width, height)
        return out


class Gating(nn.Module):
    def __init__(self, input_dim):
        super(Gating, self).__init__()
        self.proj_a = nn.Conv2d(input_dim // 2, input_dim, kernel_size=1)
        self.proj_b = nn.Conv2d(input_dim // 2, input_dim, kernel_size=1)
        self.gate = nn.Conv2d(input_dim, 1, kernel_size=1)

    def forward(self, a, b):
        a_proj = self.proj_a(a)
        b_proj = self.proj_b(b)
        gate_a = torch.sigmoid(self.gate(a_proj))
        gate_b = torch.sigmoid(self.gate(b_proj))
        gated_output = gate_a * a_proj + gate_b * b_proj
        return gated_output


class GambaBlock(nn.Module):
    def __init__(self, planes, height, width, heads=4):
        super().__init__()
        self.ggnn = GambaCell(planes // 2, height, width)
        self.mhsa = MHSA(planes // 2, width, height, head=heads)
        self.gating = Gating(input_dim=planes)

    def forward(self, x):
        n_batch, n_dims, width, height = x.size()
        half_dims = n_dims // 2
        x1 = x[:, :half_dims, :, :]
        x2 = x[:, half_dims:, :, :]
        x_ggnn = self.ggnn(x1)
        x_mhsa = self.mhsa(x2)
        x_out = self.gating(x_ggnn, x_mhsa)
        return x_out


class FourStageFeatureExtractor(nn.Module):
    def __init__(self, encoder_dims, channel_first, norm_layer, **kwargs):
        super().__init__()


        self.st_block_1 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims, out_channels=32),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=32, drop_path=0.1, norm_layer=norm_layer, **kwargs),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.st_block_2 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=32, out_channels=64),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=64, drop_path=0.1, norm_layer=norm_layer, **kwargs),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.st_block_3 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=64, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, **kwargs),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.st_block_4 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=128, out_channels=256),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=256, drop_path=0.1, norm_layer=norm_layer, **kwargs),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=480, out_channels=1024),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

    def forward(self, x):

        stage_1_out = self.st_block_1(x)

        stage_2_out = self.st_block_2(stage_1_out)

        stage_3_out = self.st_block_3(stage_2_out)

        stage_4_out = self.st_block_4(stage_3_out)

        merged_out = torch.cat([stage_1_out, stage_2_out, stage_3_out, stage_4_out], dim=1)
        final_out = self.final_conv(merged_out)

        return final_out


class Dec_last(nn.Module):
    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super().__init__()
        self.st_block_1 = nn.Sequential(
            nn.Conv2d(kernel_size=1,
                      in_channels=1856, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs[
                    'ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs[
                    'ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs[
                    'mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_2 = nn.Sequential(
            nn.Conv2d(kernel_size=1,
                      in_channels=encoder_dims, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs[
                    'ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs[
                    'ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs[
                    'mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),

        )
        self.st_block_3 = nn.Sequential(
            nn.Conv2d(kernel_size=1,
                      in_channels=encoder_dims, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs[
                    'ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs[
                    'ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs[
                    'mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.conv = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
                                  nn.BatchNorm2d(128), nn.ReLU())
        self.fdfem = FD_FEM(in_ch=128)

    def forward(self, pre_feat, post_feat, dense_embeddings):
        # print(post_feat.shape)
        size = pre_feat.size()[2:]
        dense_embeddings = F.interpolate(dense_embeddings, size, mode='bilinear', align_corners=True)

        p_1 = self.st_block_1(torch.cat([pre_feat, post_feat, dense_embeddings], dim=1))
        B, C, H, W = pre_feat.size()
        p_tensor_2 = torch.empty(B, C, H, 2 * W).cuda()
        p_tensor_2[:, :, :, ::2] = pre_feat
        p_tensor_2[:, :, :, 1::2] = post_feat
        p_2 = self.st_block_2(p_tensor_2)

        p_tensor_3 = torch.empty(B, C, H, 2 * W).cuda()
        p_tensor_3[:, :, :, 0:W] = pre_feat
        p_tensor_3[:, :, :, W:] = post_feat
        p_3 = self.st_block_3(p_tensor_3)

        p_last = self.conv(torch.cat(
            [p_1, p_2[:, :, :, ::2], p_2[:, :, :, 1::2], p_3[:, :, :, 0:W], p_3[:, :, :, W:]], dim=1))
        p_last = self.fdfem(p_last)
        return p_last

def create_positional_encoding(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term[:d_model // 2])
    pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

    if d_model % 2 != 0:
        pe[:, -1] = torch.sin(position * div_term[d_model // 2])

    return pe

class Dec_module(nn.Module):
    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super().__init__()

        self.st_block_1 = nn.Sequential(
            nn.Conv2d(kernel_size=1,
                      in_channels=encoder_dims * 2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs[
                    'ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs[
                    'ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs[
                    'mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_2 = nn.Sequential(
            nn.Conv2d(kernel_size=1,
                      in_channels=encoder_dims, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs[
                    'ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs[
                    'ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs[
                    'mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_3 = nn.Sequential(
            nn.Conv2d(kernel_size=1,
                      in_channels=encoder_dims, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs[
                    'ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs[
                    'ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs[
                    'mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.conv_1 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
                                    nn.BatchNorm2d(128), nn.ReLU())

        self.conv_2 = FusionModule(
            in_channels=128, out_channels=128, stride=1)
        self.fdfem = FD_FEM(in_ch=128)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_feat, post_feat, prior):
        p_1 = self.st_block_1(torch.cat([pre_feat, post_feat], dim=1))
        B, C, H, W = pre_feat.size()
        p_tensor_2 = torch.empty(B, C, H, 2 * W).cuda()
        p_tensor_2[:, :, :, ::2] = pre_feat
        p_tensor_2[:, :, :, 1::2] = post_feat
        p_2 = self.st_block_2(p_tensor_2)

        p_tensor_3 = torch.empty(B, C, H, 2 * W).cuda()
        p_tensor_3[:, :, :, 0:W] = pre_feat
        p_tensor_3[:, :, :, W:] = post_feat
        p_3 = self.st_block_2(p_tensor_3)

        p_last = self.conv_1(torch.cat(
            [p_1, p_2[:, :, :, ::2], p_2[:, :, :, 1::2], p_3[:, :, :, 0:W], p_3[:, :, :, W:]], dim=1))

        p_last = self._upsample_add(prior, p_last)
        p_last = self.conv_2(p_last)
        p_last = self.fdfem(p_last)
        return p_last


class FFM(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(FFM, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(out_planes)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv1(x2)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)
        x_x = x1 * x2
        x_x = self.relu(x_x)
        x_x = x_x + x2
        x_x = x_x * x1
        x_x = self.relu(x_x)
        return x_x


class GMM(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GMM, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):
        embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
        norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        gate = 1 + torch.sigmoid(embedding * norm + self.beta)
        return x * gate


class SqueezeDoubleConvOld(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SqueezeDoubleConvOld, self).__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.double_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.acfun = nn.GELU()
        self.gmm = GMM(out_channels)

    def forward(self, x):
        x = self.squeeze(x)
        # print(f"Squeeze output shape: {x.shape}")  # Debugging
        x = self.gmm(x)
        block_x = self.double_conv(x)
        # print(f"Double conv output shape: {block_x.shape}")  # Debugging
        x = self.acfun(x + block_x)
        # print(f"Output after addition and activation: {x.shape}")  # Debugging
        return x


class CDDecoder(nn.Module):
    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super(CDDecoder, self).__init__()

        self.dec1 = Dec_last(encoder_dims=encoder_dims[-1], channel_first=channel_first, norm_layer=norm_layer,
                             ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer, **kwargs)
        self.dec2 = Dec_module(encoder_dims=encoder_dims[-2], channel_first=channel_first, norm_layer=norm_layer,
                               ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer, **kwargs)
        self.dec3 = Dec_module(encoder_dims=encoder_dims[-3], channel_first=channel_first, norm_layer=norm_layer,
                               ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer, **kwargs)
        self.dec4 = Dec_module(encoder_dims=encoder_dims[-4], channel_first=channel_first, norm_layer=norm_layer,
                               ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer, **kwargs)
        self.ffm = FFM(896 , 896 )
        self.decoder = nn.Sequential(SqueezeDoubleConvOld(896, 64), nn.Conv2d(64, 1, 1))

        self.prompt_encoder = PromptEncoder(
            embed_dim=64,
            image_embedding_size=(64, 64),
            input_image_size=(256, 256),
            mask_in_chans=16,
        )

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_features, post_features):
        a = torch.randn(1, 3, 256, 256)
        size = a.size()[2:]
        pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features

        post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features

        layer_ss = self.ffm(pre_feat_4, post_feat_4)
        feature_fuse = layer_ss

        change_map = self.decoder(feature_fuse)

        change_map1 = F.interpolate(change_map, size, mode='bilinear', align_corners=True)
        self.weight_net = nn.Sigmoid()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_net = self.weight_net.to(device)

        change_map2 = change_map1.squeeze(1).detach().cpu().numpy()  #
        boxes_list = []


        for i in range(change_map2.shape[0]):

            img = change_map2[i]

            contours, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            boxes = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append([x, y, x + w, y + h])  # [x1, y1, x2, y2]

            if boxes:  # 确保有有效的边界框
                boxes_list.append(torch.tensor(boxes, dtype=torch.float32).to(device))
            else:
                boxes_list.append(torch.tensor([], dtype=torch.float32).to(device))
        self.weight_net = nn.Sigmoid().to(device)
        for i in range(len(boxes_list)):
            boxes = boxes_list[i]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=boxes,
                masks=change_map1
            )
        change_map = F.interpolate(change_map, dense_embeddings.size()[2:], mode='bilinear', align_corners=True)
        dense_embeddings=dense_embeddings*self.weight_net(change_map )
        p4 = self.dec1(pre_feat_4, post_feat_4, dense_embeddings)
        p3 = self.dec2(pre_feat_3, post_feat_3, p4)
        p2 = self.dec3(pre_feat_2, post_feat_2, p3)
        p1 = self.dec4(pre_feat_1, post_feat_1, p2)

        return p1


if __name__ == '__main__':
    x = torch.randn(1, 4, 128, 128).cuda()
    model = PartFusionModule(in_ch=4).cuda()
    y = model(x)
    print(y.shape)
