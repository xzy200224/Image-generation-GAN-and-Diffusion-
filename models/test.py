from Unet import Unet
import torch
import torch.nn as nn
import torch.nn.functional as F

# 创建 Unet 实例
input_channels = 3
output_channels = 1
basedim = 8
downdeepth = 4
model_type = '2d'
isresunet = True
use_max_pool = True
use_avg_pool = False
istranspose = True
norm_type = 'BN'
activation_function = 'sigmoid'

unet_model = Unet(input_channels, output_channels, basedim, downdeepth, model_type, isresunet,
                  use_max_pool, use_avg_pool, istranspose, norm_type, activation_function)

# 生成测试数据
input_data = torch.randn(2, 3, 256, 256)  # 批量大小=2，通道数=3，尺寸=256x256

# 前向传播
output_cls, output_prob = unet_model(input_data)

# 打印输出尺寸
print(f"Output classification shape: {output_cls.shape}")
print(f"Output probability shape: {output_prob.shape}")