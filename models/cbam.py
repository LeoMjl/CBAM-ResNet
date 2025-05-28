import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    通道注意力模块
    使用全局平均池化和最大池化提取通道特征，然后通过共享的MLP学习通道注意力权重
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        # 使用较小的隐藏层来减少参数数量
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享的MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return torch.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    空间注意力模块
    使用通道维度上的平均池化和最大池化提取空间特征，然后通过卷积学习空间注意力权重
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 确保kernel_size是奇数，以便于padding
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        # 卷积层用于学习空间注意力图
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        
    def forward(self, x):
        # 沿着通道维度计算平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 拼接平均池化和最大池化的结果
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        
        return torch.sigmoid(out)


class CBAM(nn.Module):
    """
    CBAM注意力模块
    结合通道注意力和空间注意力，按顺序应用
    """
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        # 先应用通道注意力
        x = x * self.channel_attention(x)
        # 再应用空间注意力
        x = x * self.spatial_attention(x)
        return x