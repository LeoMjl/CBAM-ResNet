import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义基本的残差块
class BasicBlock(nn.Module):
    """
    ResNet的基本构建块
    包含两个3x3的卷积层和一个跳跃连接
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        # 第一个卷积层，可能改变特征图大小
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 第二个卷积层，保持特征图大小不变
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入和输出通道数不同，需要使用1x1卷积进行调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 添加跳跃连接
        out = F.relu(out)
        return out


# 定义ResNet模型
class ResNet(nn.Module):
    """
    ResNet模型
    可以根据不同的层数配置构建不同深度的ResNet
    """
    def __init__(self, block, num_blocks, num_classes=50):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 构建ResNet的四个阶段，每个阶段包含多个残差块
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # 全连接层用于分类
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        构建包含多个残差块的层
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # 使用自适应平均池化替代固定大小的平均池化
        out = F.adaptive_avg_pool2d(out, 1)  # 输出大小为1x1
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# 定义不同深度的ResNet模型
def ResNet18(num_classes=50):
    """
    构建ResNet-18模型
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)