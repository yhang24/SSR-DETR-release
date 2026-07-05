import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------
# 1. UWT 版可学习小波变换（不下采样）
# -------------------------------
class LearnableWaveletTransform(nn.Module):
    """
    可学习小波变换（UWT，不下采样）
    - 输入:  x  [B, C, H, W]
    - 输出:  LL, HL, LH, HH  （形状都为 [B, C, H, W]）
    """
    def __init__(self, in_channels, wavelet_type='haar', learnable=True):
        super(LearnableWaveletTransform, self).__init__()
        self.in_channels = in_channels
        self.learnable = learnable

        # 初始化 Haar 小波滤波器
        if wavelet_type == 'haar':
            low_filter = torch.tensor([[0.5, 0.5],
                                       [0.5, 0.5]], dtype=torch.float32)
            high_filter_h = torch.tensor([[0.5, -0.5],
                                          [0.5, -0.5]], dtype=torch.float32)
            high_filter_v = torch.tensor([[0.5, 0.5],
                                          [-0.5, -0.5]], dtype=torch.float32)
            high_filter_d = torch.tensor([[0.5, -0.5],
                                          [-0.5, 0.5]], dtype=torch.float32)
        else:
            raise NotImplementedError(f"Wavelet type {wavelet_type} not implemented")

        if self.learnable:
            # UWT: stride=1，不下采样。padding 在 forward 里手动做。
            stride = 1
            padding = 0

            self.low_pass = nn.Conv2d(
                in_channels, in_channels, kernel_size=2, stride=stride,
                padding=padding, groups=in_channels, bias=False
            )
            self.high_pass_h = nn.Conv2d(
                in_channels, in_channels, kernel_size=2, stride=stride,
                padding=padding, groups=in_channels, bias=False
            )
            self.high_pass_v = nn.Conv2d(
                in_channels, in_channels, kernel_size=2, stride=stride,
                padding=padding, groups=in_channels, bias=False
            )
            self.high_pass_d = nn.Conv2d(
                in_channels, in_channels, kernel_size=2, stride=stride,
                padding=padding, groups=in_channels, bias=False
            )

            self.init_wavelet_filters(low_filter, high_filter_h, high_filter_v, high_filter_d)
        else:
            # 固定滤波器
            self.register_buffer('low_filter', low_filter.unsqueeze(0).unsqueeze(0))
            self.register_buffer('high_filter_h', high_filter_h.unsqueeze(0).unsqueeze(0))
            self.register_buffer('high_filter_v', high_filter_v.unsqueeze(0).unsqueeze(0))
            self.register_buffer('high_filter_d', high_filter_d.unsqueeze(0).unsqueeze(0))

    def init_wavelet_filters(self, low_filter, high_filter_h, high_filter_v, high_filter_d):
        """初始化可学习的小波滤波器为 Haar 系数"""
        with torch.no_grad():
            for i in range(self.in_channels):
                self.low_pass.weight[i, 0] = low_filter
                self.high_pass_h.weight[i, 0] = high_filter_h
                self.high_pass_v.weight[i, 0] = high_filter_v
                self.high_pass_d.weight[i, 0] = high_filter_d

    @staticmethod
    def _uwt_pad(x):
        """
        为了保持输出大小与输入一致，对右和下做 1 像素反射 padding。
        输入:  B,C,H,W
        输出:  B,C,H+1,W+1  -> kernel=2, stride=1, valid 卷积后得到 H,W
        """
        return F.pad(x, (0, 1, 0, 1), mode='reflect')  # (left, right, top, bottom)

    def forward(self, x):
        """
        前向传播 - UWT 小波分解
        返回的四个子带与输入 x 在空间尺寸上完全一致
        """
        if self.learnable:
            x_pad = self._uwt_pad(x)
            LL = self.low_pass(x_pad)
            HL = self.high_pass_h(x_pad)
            LH = self.high_pass_v(x_pad)
            HH = self.high_pass_d(x_pad)
        else:
            x_pad = self._uwt_pad(x)
            C = x.size(1)
            LL = F.conv2d(x_pad, self.low_filter.expand(C, -1, -1, -1),
                          groups=C, stride=1)
            HL = F.conv2d(x_pad, self.high_filter_h.expand(C, -1, -1, -1),
                          groups=C, stride=1)
            LH = F.conv2d(x_pad, self.high_filter_v.expand(C, -1, -1, -1),
                          groups=C, stride=1)
            HH = F.conv2d(x_pad, self.high_filter_d.expand(C, -1, -1, -1),
                          groups=C, stride=1)
        return LL, HL, LH, HH


# -------------------------------
# 2. HiLo 注意力 + UWT 小波
# -------------------------------
class WaveletHiLo(nn.Module):
    """
    结合 UWT 可学习小波变换的 HiLo Attention
    - 只做小波分解，不做逆变换
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., window_size=2, alpha=0.5,
                 use_wavelet=True, learnable_wavelet=True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim / num_heads)
        self.dim = dim
        self.use_wavelet = use_wavelet
        self.learnable_wavelet = learnable_wavelet

        # 小波相关（只保留 UWT 分解）
        if self.use_wavelet:
            self.dwt = LearnableWaveletTransform(dim, learnable=learnable_wavelet)
            # 处理高低频子带
            self.high_freq_conv = nn.Conv2d(dim * 3, dim, 1)  # HL, LH, HH
            self.low_freq_conv = nn.Conv2d(dim, dim, 1)       # LL

        # Lo-Fi / Hi-Fi 维度拆分
        self.l_heads = int(num_heads * alpha)
        self.l_dim = self.l_heads * head_dim

        self.h_heads = num_heads - self.l_heads
        self.h_dim = self.h_heads * head_dim

        self.ws = window_size

        if self.ws == 1:
            # ws == 1 时退化为标准 MHSA
            self.h_heads = 0
            self.h_dim = 0
            self.l_heads = num_heads
            self.l_dim = dim

        self.scale = qk_scale or head_dim ** -0.5

        # Lo-Fi 注意力
        if self.l_heads > 0:
            if self.ws != 1:
                self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
            self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
            self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
            self.l_proj = nn.Linear(self.l_dim, self.l_dim)

        # Hi-Fi 注意力
        if self.h_heads > 0:
            self.h_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)
            self.h_proj = nn.Linear(self.h_dim, self.h_dim)

    def hifi(self, x):
        """高频注意力机制（局部窗口）"""
        B, H, W, C = x.shape
        # assert H % self.ws == 0 and W % self.ws == 0, "H,W 必须能被 window_size 整除"
        h_group, w_group = H // self.ws, W // self.ws
        total_groups = h_group * w_group

        # [B, H, W, C] -> [B, h_group, ws, w_group, ws, C] -> 分组
        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)

        # 计算 qkv
        qkv = self.h_qkv(x).reshape(
            B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads
        ).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, G, heads, ws*ws, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(
            B, h_group, w_group, self.ws, self.ws, self.h_dim
        )
        x = attn.transpose(2, 3).reshape(B, H, W, self.h_dim)

        x = self.h_proj(x)
        return x

    def lofi(self, x):
        """低频注意力机制（全局）"""
        B, H, W, C = x.shape

        q = self.l_q(x).reshape(
            B, H * W, self.l_heads, self.l_dim // self.l_heads
        ).permute(0, 2, 1, 3)  # [B, heads, HW, dim_per_head]

        if self.ws > 1:
            x_ = x.permute(0, 3, 1, 2)  # [B,C,H,W]
            x_ = self.sr(x_)            # 下采样的 key/value
            x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
            kv = self.l_kv(x_).reshape(
                B, -1, 2, self.l_heads, self.l_dim // self.l_heads
            ).permute(2, 0, 3, 1, 4)
        else:
            kv = self.l_kv(x).reshape(
                B, -1, 2, self.l_heads, self.l_dim // self.l_heads
            ).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
        x = self.l_proj(x)
        return x

    def wavelet_decompose(self, x):
        """UWT 小波分解"""
        if not self.use_wavelet:
            return x, None, None, None
        LL, HL, LH, HH = self.dwt(x)
        return LL, HL, LH, HH

    def forward(self, x):
        """
        x: [B, C, H, W]
        返回: [B, C', H, W]，其中 C' = dim（当 h,l 都有时为 dim）
        """
        # 1) 小波分解
        if self.use_wavelet:
            LL, HL, LH, HH = self.wavelet_decompose(x)
            # UWT: 子带与 x 同尺寸，不需要任何上采样

            # 低频输入
            low_freq_input = self.low_freq_conv(LL).permute(0, 2, 3, 1)  # [B,H,W,C]
            # 高频输入
            high_freq_combined = torch.cat([HL, LH, HH], dim=1)         # [B,3C,H,W]
            high_freq_input = self.high_freq_conv(high_freq_combined).permute(0, 2, 3, 1)
        else:
            low_freq_input = high_freq_input = x.permute(0, 2, 3, 1)

        # 2) Hi-Lo 注意力
        if self.h_heads == 0:
            out = self.lofi(low_freq_input)
            return out.permute(0, 3, 1, 2)

        if self.l_heads == 0:
            out = self.hifi(high_freq_input)
            return out.permute(0, 3, 1, 2)

        hifi_out = self.hifi(high_freq_input)
        lofi_out = self.lofi(low_freq_input)

        # 3) 通道拼接
        combined_out = torch.cat((hifi_out, lofi_out), dim=-1)  # [B,H,W,h_dim+l_dim=dim]
        combined_out = combined_out.permute(0, 3, 1, 2)         # [B,dim,H,W]
        return combined_out


# -------------------------------
# 3. 你原来的辅助模块（Conv / LayerNorm / WDFA）
# -------------------------------
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
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


class LayerNorm(nn.Module):
    """
    LayerNorm that supports channels_last or channels_first.
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
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


class WDFA(nn.Module):
    """
    结合 UWT 小波变换的 transformer encoder 层
    - 内部使用上面的 WaveletHiLo（只做小波分解，不做逆变换）
    """

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(),
                 normalize_before=False, use_wavelet=True, learnable_wavelet=True, alpha=0.5):
        super().__init__()
        self.Attention = WaveletHiLo(
            c1, num_heads=num_heads, use_wavelet=use_wavelet,
            learnable_wavelet=learnable_wavelet, alpha=alpha
        )
        self.fc1 = nn.Conv2d(c1, cm, 1)
        self.fc2 = nn.Conv2d(cm, c1, 1)

        self.norm1 = LayerNorm(c1)
        self.norm2 = LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """post-norm"""
        src2 = self.Attention(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


# -------------------------------
# 4. 简单测试样例
# -------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, C, H, W = 2, 64, 32, 32
    x = torch.randn(B, C, H, W)

    print("=== 测试 WaveletHiLo(UWT) ===")
    whl = WaveletHiLo(dim=C, num_heads=8, window_size=2, alpha=0.5,
                      use_wavelet=True, learnable_wavelet=True)
    y = whl(x)
    print("输入形状:", x.shape)
    print("输出形状:", y.shape)

    loss = y.mean()
    loss.backward()
    print("WaveletHiLo 反向传播 OK")

    print("\n=== 测试 WDFA ===")
    aifi = WDFA(c1=C, cm=128, num_heads=8, dropout=0.1,
                            use_wavelet=True, learnable_wavelet=True, alpha=0.5)
    y2 = aifi(x)
    print("AIFI 输出形状:", y2.shape)

    loss2 = y2.mean()
    loss2.backward()
    print("WDFA 反向传播 OK")
