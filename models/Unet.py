import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

def extend_by_dim(krnlsz, model_type='3d', half_dim=1):
    if model_type == '2d':
        outsz = [krnlsz] * 2
    elif model_type == '3d':
        outsz = [krnlsz] * 3
    elif model_type == '2.5d':
        outsz = [krnlsz] * 2 + [(np.array(krnlsz) * 0 + 1) * half_dim]
    else:
        outsz = [krnlsz]
    return tuple(outsz)


def build_end_activation(input, activation='linear', alpha=None):
    if activation == 'softmax':
        output = F.softmax(input, dim=1)
    elif activation == 'sigmoid':
        output = torch.sigmoid(input)
    elif activation == 'elu':
        if alpha is None: alpha = 0.01
        output = F.elu(input, alpha=alpha)
    elif activation == 'lrelu':
        if alpha is None: alpha = 0.01
        output = F.leaky_relu(input, negative_slope=alpha)
    elif activation == 'relu':
        output = F.relu(input)
    elif activation == 'tanh':
        output = F.tanh(input)
    else:
        output = input
    return output

class BasicConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding='same', mid_channel=None,
                 model_type='2d', norm_type='bn', residualskip=False):
        super(BasicConvBlock, self).__init__()
        self.change_dimension = input_channel != output_channel
        self.model_type = model_type
        self.norm_type = norm_type
        self.residualskip = residualskip
        padding = {'same': kernel_size // 2, 'valid': 0}[padding] if padding in ['same', 'valid'] else padding
        mid_channel = output_channel if mid_channel is None else mid_channel

        if self.model_type == '3d' and self.norm_type == 'in':
            self.ConvBlock, self.NormBlock = nn.Conv3d, nn.InstanceNorm3d
        elif self.model_type == '3d' and self.norm_type == 'bn':
            self.ConvBlock, self.NormBlock = nn.Conv3d, nn.BatchNorm3d
        elif self.model_type == '2d' and self.norm_type == 'in':
            self.ConvBlock, self.NormBlock = nn.Conv2d, nn.InstanceNorm2d
        else:
            self.ConvBlock, self.NormBlock = nn.Conv2d, nn.BatchNorm2d

        self.build_block(input_channel, mid_channel, output_channel, kernel_size, stride, padding)

    def build_block(self, input_channel, mid_channel, output_channel, kernel_size, stride, padding):
        self.short_cut_conv = self.ConvBlock(input_channel, output_channel, 1,
                                             self.extdim(stride))
        self.conv1 = self.ConvBlock(input_channel, mid_channel, self.extdim(kernel_size),
                                   self.extdim(stride), padding=self.extdim(padding),
                                    padding_mode='reflect')
        self.conv2 = self.ConvBlock(mid_channel, output_channel, self.extdim(kernel_size),
                                    self.extdim(1), padding=self.extdim(padding),
                                    padding_mode='reflect')
        if self.norm_type == 'in':
            self.norm0 = self.NormBlock(output_channel, affine=True, track_running_stats=True)
            self.norm1 = self.NormBlock(mid_channel, affine=True, track_running_stats=True)
            self.norm2 = self.NormBlock(output_channel, affine=True, track_running_stats=True)
        else:
            self.norm0 = self.NormBlock(output_channel)
            self.norm1 = self.NormBlock(mid_channel)
            self.norm2 = self.NormBlock(output_channel)
        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()

    def extdim(self, krnlsz, model_type='2d'):
        if model_type == '2d':
            outsz = [krnlsz] * 2
        elif model_type == '3d':
            outsz = [krnlsz] * 3
        else:
            outsz = [krnlsz]
        return tuple(outsz)

    def forward(self, x):
        if self.residualskip and self.change_dimension:
            short_cut_conv = self.norm0(self.short_cut_conv(x))
        else:
            short_cut_conv = x
        o_c1 = self.relu1(self.norm1(self.conv1(x)))
        o_c2 = self.norm2(self.conv2(o_c1))
        if self.residualskip:
            out = self.relu2(o_c2+short_cut_conv)
        else:
            out = self.relu2(o_c2)
        return out


class Unet(nn.Module):
    def __init__(self, input_channel, output_channel, basedim=8, downdeepth=4, model_type='2D', isresunet=True,
                use_max_pool=True, use_avg_pool=False, istranspose=False, norm_type='BN', activation_function='sigmoid',
                istransunet=False):
        super(Unet, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.model_type = model_type.lower()
        self.norm_type = norm_type.lower()
        self.basedim = basedim
        self.downdeepth = downdeepth
        self.activation_function = activation_function
        self.isresunet = isresunet
        self.istransunet = istransunet
        self.istranspose = istranspose
        self.use_max_pool = use_max_pool
        self.use_avg_pool = use_avg_pool
        # 根据类型选择卷积和归一化
        if self.model_type == '3d' and self.norm_type == 'in':
            self.ConvBlock, self.NormBlock = nn.Conv3d, nn.InstanceNorm3d
        elif self.model_type == '3d' and self.norm_type == 'bn':
            self.ConvBlock, self.NormBlock = nn.Conv3d, nn.BatchNorm3d
        elif self.model_type == '2d' and self.norm_type == 'in':
            self.ConvBlock, self.NormBlock = nn.Conv2d, nn.InstanceNorm2d
        else:
            self.ConvBlock, self.NormBlock = nn.Conv2d, nn.BatchNorm2d
        if self.model_type == '3d':
            self.MaxPool, self.AvgPool, self.ConvTranspose = nn.MaxPool3d, nn.AvgPool3d, nn.ConvTranspose3d
        else:
            self.MaxPool, self.AvgPool, self.ConvTranspose = nn.MaxPool2d, nn.AvgPool2d, nn.ConvTranspose2d
        self.build_network(input_channel, basedim, downdeepth, output_channel)

    def extdim(self, krnlsz, model_type='2d'):
        if model_type == '2d':
            outsz = [krnlsz] * 2
        elif model_type == '3d':
            outsz = [krnlsz] * 3
        else:
            outsz = [krnlsz]
        return tuple(outsz)

    def build_network(self, in_channels, basedim, downdeepth=4, output_channel=1):
        self.begin_conv = BasicConvBlock(in_channels, basedim, 3, 1, model_type=self.model_type,
                                         residualskip=self.isresunet)
        if self.use_max_pool:
            self.encoding_block = nn.ModuleList([nn.Sequential(
                self.MaxPool(self.extdim(3), self.extdim(2), padding=self.extdim(1)),
                BasicConvBlock(basedim * 2 ** convidx, basedim * 2 ** (convidx + 1), 3, 1,
                               model_type=self.model_type, residualskip=self.isresunet)) for
                convidx in range(0, downdeepth)])
        elif self.use_avg_pool:
            self.encoding_block = nn.ModuleList([nn.Sequential(
                self.AvgPool(self.extdim(3), self.extdim(2), padding=self.extdim(1)),
                BasicConvBlock(basedim * 2 ** convidx, basedim * 2 ** (convidx + 1), 3, 1,
                               model_type=self.model_type, residualskip=self.isresunet)) for
                convidx in range(0, downdeepth)])
        else:
            self.encoding_block = nn.ModuleList([nn.Sequential(
                BasicConvBlock(basedim * 2 ** convidx, basedim * 2 ** (convidx + 1), 3, 2,
                               model_type=self.model_type, residualskip=self.isresunet)) for
                convidx in range(0, downdeepth)])

        trans_dim = basedim * 2 ** downdeepth
        if self.istransunet:
            self.trans_block = nn.Sequential(nn.TransformerEncoder(nn.TransformerEncoderLayer(trans_dim, 8), 2))
        else:
            self.trans_block = nn.Sequential(
                BasicConvBlock(trans_dim, trans_dim, 1, 1, model_type=self.model_type, residualskip=self.isresunet),
                BasicConvBlock(trans_dim, trans_dim, 1, 1, model_type=self.model_type, residualskip=self.isresunet),
            )

        if self.istranspose:
            self.up_block = nn.ModuleList([
                self.ConvTranspose(basedim * 2 ** (convidx + 1), basedim * 2 ** convidx, 4, 2, padding=1) for convidx in
                range(0, downdeepth)
            ])
            self.decoding_block = nn.ModuleList([
                BasicConvBlock(basedim * 2 ** (convidx + 1), basedim * 2 ** convidx, 3, 1, model_type=self.model_type,
                               mid_channel=basedim * 2 ** (convidx + 1), residualskip=self.isresunet) for convidx in
                range(0, downdeepth)
            ])
        else:
            self.decoding_block = nn.ModuleList([
                BasicConvBlock(basedim * 2 ** (convidx + 2), basedim * 2 ** convidx, 3, 1, model_type=self.model_type,
                               mid_channel=basedim * 2 ** (convidx + 1), residualskip=self.isresunet) for convidx in
                range(0, downdeepth)
            ])
            self.end_conv = BasicConvBlock(basedim * 2, basedim, 3, 1, model_type=self.model_type,
                                           residualskip=self.isresunet)

        self.class_conv = self.ConvBlock(basedim, output_channel, self.extdim(3), stride=1,
                                         padding=self.extdim(1), padding_mode='reflect')

    def forward(self, x):
        o_c1 = self.begin_conv(x)
        feats = [o_c1, ]
        for convidx in range(0, len(self.encoding_block)):
            o_c1 = self.encoding_block[convidx](o_c1)
            feats.append(o_c1)

        if self.istransunet:
            o_c2 = torch.transpose(o_c1.view([*o_c1.size()[0:2], -1]), 1, 2)
            o_c2 = self.trans_block(o_c2)
            o_c2 = torch.transpose(o_c2, 1, 2).view(o_c1.size())
        else:
            o_c2 = self.trans_block(o_c1)

        if self.istranspose:
            for convidx in range(self.downdeepth, 0, -1):
                o_c2 = self.up_block[convidx - 1](o_c2)
                o_c2 = torch.concat((o_c2, feats[convidx - 1]), dim=1)
                o_c2 = self.decoding_block[convidx - 1](o_c2)
            o_c3 = o_c2
        else:
            for convidx in range(self.downdeepth, 0, -1):
                o_c2 = torch.concat((o_c2, feats[convidx]), dim=1)
                o_c2 = self.decoding_block[convidx - 1](o_c2)
                if self.model_type == "3d":
                    o_c2 = F.interpolate(o_c2, scale_factor=self.extdim(2), mode="trilinear")
                else:
                    o_c2 = F.interpolate(o_c2, scale_factor=self.extdim(2), mode="bilinear")
            o_c3 = self.end_conv(torch.concat((o_c2, feats[0]), dim=1))

        o_cls = self.class_conv(o_c3)
        prob = build_end_activation(o_cls, self.activation_function)
        return [o_cls, prob, ]