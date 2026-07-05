import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import warnings
import numpy as np

__all__ = ['FAFF']


def xavier_init(module: nn.Module,
                gain: float = 1,
                bias: float = 0,
                distribution: str = 'normal') -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def carafe(x, normed_mask, kernel_size, group=1, up=1):
    b, c, h, w = x.shape
    _, m_c, m_h, m_w = normed_mask.shape
    assert m_h == up * h
    assert m_w == up * w
    pad = kernel_size // 2
    pad_x = F.pad(x, pad=[pad] * 4, mode='reflect')
    unfold_x = F.unfold(pad_x, kernel_size=(kernel_size, kernel_size), stride=1, padding=0)
    unfold_x = unfold_x.reshape(b, c * kernel_size * kernel_size, h, w)
    unfold_x = F.interpolate(unfold_x, scale_factor=up, mode='nearest')
    unfold_x = unfold_x.reshape(b, c, kernel_size * kernel_size, m_h, m_w)
    normed_mask = normed_mask.reshape(b, 1, kernel_size * kernel_size, m_h, m_w)
    res = unfold_x * normed_mask
    res = res.sum(dim=2).reshape(b, c, m_h, m_w)
    return res


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def hamming2D(M, N):
    """
    Generate 2D Hamming window (M x N).
    """
    hamming_x = np.hamming(M)
    hamming_y = np.hamming(N)
    return np.outer(hamming_x, hamming_y)


class FAFF(nn.Module):
    def __init__(self,
                 channels,
                 scale_factor=1,
                 lowpass_kernel=5,
                 highpass_kernel=3,
                 up_group=1,
                 encoder_kernel=3,
                 encoder_dilation=1,
                 compressed_channels=64,
                 align_corners=False,
                 upsample_mode='nearest',
                 comp_feat_upsample=True,   # use ALPF & AHPF for init upsampling
                 use_high_pass=True,
                 use_low_pass=True,
                 hr_residual=True,
                 semi_conv=True,
                 hamming_window=True,       # for regularization
                 both_high_pass=False,      # NEW: if True, both branches use high-pass filters
                 **kwargs):
        super().__init__()
        hr_channels, lr_channels = channels
        self.scale_factor = scale_factor
        self.lowpass_kernel = lowpass_kernel
        self.highpass_kernel = highpass_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation

        # NOTE: kept as in your original code (overrides passed-in compressed_channels)
        self.compressed_channels = (hr_channels + lr_channels) // 8

        self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels, 1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels, 1)

        # Low-pass mask generator (ALPF generator)
        self.content_encoder = nn.Conv2d(
            self.compressed_channels,
            lowpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1
        )

        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.hr_residual = hr_residual
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass
        self.semi_conv = semi_conv
        self.comp_feat_upsample = comp_feat_upsample

        # NEW flag
        self.both_high_pass = both_high_pass
        if self.both_high_pass and (not self.use_high_pass):
            raise ValueError("both_high_pass=True requires use_high_pass=True (content_encoder2 must exist).")

        # PixelShuffle upsample branch for LR features
        self.lr_channel_expander = nn.Conv2d(
            lr_channels,
            lr_channels * (scale_factor ** 2),
            kernel_size=3,
            padding=1
        )

        # High-pass mask generator (AHPF generator)
        if self.use_high_pass:
            self.content_encoder2 = nn.Conv2d(
                self.compressed_channels,
                highpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
                self.encoder_kernel,
                padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation,
                groups=1
            )

        self.hamming_window = hamming_window
        lowpass_pad = 0
        highpass_pad = 0
        if self.hamming_window:
            self.register_buffer(
                'hamming_lowpass',
                torch.FloatTensor(hamming2D(lowpass_kernel + 2 * lowpass_pad,
                                            lowpass_kernel + 2 * lowpass_pad))[None, None,]
            )
            self.register_buffer(
                'hamming_highpass',
                torch.FloatTensor(hamming2D(highpass_kernel + 2 * highpass_pad,
                                            highpass_kernel + 2 * highpass_pad))[None, None,]
            )
        else:
            self.register_buffer('hamming_lowpass', torch.FloatTensor([1.0]))
            self.register_buffer('hamming_highpass', torch.FloatTensor([1.0]))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.content_encoder, std=0.001)
        xavier_init(self.lr_channel_expander, distribution='uniform')
        if self.use_high_pass:
            normal_init(self.content_encoder2, std=0.001)

    def kernel_normalizer(self, mask, kernel, scale_factor=None, hamming=1):
        if scale_factor is not None:
            mask = F.pixel_shuffle(mask, self.scale_factor)

        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / float(kernel ** 2))

        mask = mask.view(n, mask_channel, -1, h, w)
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)

        mask = mask.view(n, mask_channel, kernel, kernel, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)

        mask = mask * hamming
        mask /= mask.sum(dim=(-1, -2), keepdims=True)

        mask = mask.view(n, mask_channel, h, w, -1)
        mask = mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
        return mask

    def forward(self, x, use_checkpoint=False):
        hr_feat, lr_feat = x
        if use_checkpoint:
            return checkpoint(self._forward, hr_feat, lr_feat)
        else:
            return self._forward(hr_feat, lr_feat)

    def _forward(self, hr_feat, lr_feat):
        compressed_hr_feat = self.hr_channel_compressor(hr_feat)
        compressed_lr_feat = self.lr_channel_compressor(lr_feat)

        # ========== choose encoder/kernel/window for the "lp branch" ==========
        if self.both_high_pass:
            branch_encoder = self.content_encoder2
            branch_kernel = self.highpass_kernel
            branch_hamming = self.hamming_highpass
        else:
            branch_encoder = self.content_encoder
            branch_kernel = self.lowpass_kernel
            branch_hamming = self.hamming_lowpass
        # ================================================================

        if self.semi_conv:
            if self.comp_feat_upsample:
                if self.use_high_pass:
                    # High-pass on compressed HR (init)
                    mask_hr_hr_feat = self.content_encoder2(compressed_hr_feat)
                    mask_hr_init = self.kernel_normalizer(
                        mask_hr_hr_feat, self.highpass_kernel, hamming=self.hamming_highpass
                    )
                    compressed_hr_feat = compressed_hr_feat + compressed_hr_feat - carafe(
                        compressed_hr_feat,
                        mask_hr_init.to(compressed_hr_feat.dtype),
                        self.highpass_kernel,
                        self.up_group,
                        1
                    )

                    # "Low-pass branch" (or HP if both_high_pass=True) from compressed HR
                    mask_lr_hr_feat = branch_encoder(compressed_hr_feat)
                    mask_lr_init = self.kernel_normalizer(
                        mask_lr_hr_feat, branch_kernel, hamming=branch_hamming
                    )

                    # "Low-pass branch" (or HP) from compressed LR, guided by init mask
                    mask_lr_lr_feat_lr = branch_encoder(compressed_lr_feat)
                    mask_lr_lr_feat = F.interpolate(
                        carafe(
                            mask_lr_lr_feat_lr,
                            mask_lr_init.to(compressed_hr_feat.dtype),
                            branch_kernel,
                            self.up_group,
                            2
                        ),
                        size=compressed_hr_feat.shape[-2:],
                        mode='nearest'
                    )
                    mask_lr = mask_lr_hr_feat + mask_lr_lr_feat

                    # Normalize final guide mask (lp or hp depending on flag)
                    mask_lr_init = self.kernel_normalizer(
                        mask_lr, branch_kernel, hamming=branch_hamming
                    )

                    # Use the guide mask to bring LR high-pass mask contribution
                    mask_hr_lr_feat = F.interpolate(
                        carafe(
                            self.content_encoder2(compressed_lr_feat),
                            mask_lr_init.to(compressed_hr_feat.dtype),
                            branch_kernel,
                            self.up_group,
                            2
                        ),
                        size=compressed_hr_feat.shape[-2:],
                        mode='nearest'
                    )
                    mask_hr = mask_hr_hr_feat + mask_hr_lr_feat
                else:
                    raise NotImplementedError
            else:
                # No comp_feat_upsample: generate masks directly (lp branch may switch to hp)
                mask_lr = branch_encoder(compressed_hr_feat) + F.interpolate(
                    branch_encoder(compressed_lr_feat),
                    size=compressed_hr_feat.shape[-2:],
                    mode='nearest'
                )
                if self.use_high_pass:
                    mask_hr = self.content_encoder2(compressed_hr_feat) + F.interpolate(
                        self.content_encoder2(compressed_lr_feat),
                        size=compressed_hr_feat.shape[-2:],
                        mode='nearest'
                    )
        else:
            compressed_x = F.interpolate(
                compressed_lr_feat, size=compressed_hr_feat.shape[-2:], mode='nearest'
            ) + compressed_hr_feat
            mask_lr = branch_encoder(compressed_x)
            if self.use_high_pass:
                mask_hr = self.content_encoder2(compressed_x)

        # ========== LR upsampling: PixelShuffle ==========
        lr_feat_expanded = self.lr_channel_expander(lr_feat)
        lr_feat = F.pixel_shuffle(lr_feat_expanded, upscale_factor=self.scale_factor)

        if lr_feat.shape[2:] != hr_feat.shape[2:]:
            lr_feat = F.interpolate(
                lr_feat,
                size=hr_feat.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        # ================================================================

        if self.use_high_pass:
            mask_hr = self.kernel_normalizer(
                mask_hr, self.highpass_kernel, hamming=self.hamming_highpass
            )
            if self.hr_residual:
                hr_feat_hf = hr_feat - carafe(
                    hr_feat,
                    mask_hr.to(compressed_hr_feat.dtype),
                    self.highpass_kernel,
                    self.up_group,
                    1
                )
                hr_feat = hr_feat_hf + hr_feat
            else:
                hr_feat = hr_feat_hf

        return hr_feat + lr_feat
