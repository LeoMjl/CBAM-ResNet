import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cbam_resnet import CBAMResNet18
from scripts.utils import get_device
from configs.config import DATASET, AUGMENTATION, DEVICE, LOGGING


class AttentionVisualizer:
    """
    CBAM注意力机制可视化类
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        
        # 注册钩子
        self.hooks = []
        self.attention_maps = {}
        
        # 注册所有CBAM模块的钩子
        for name, module in model.named_modules():
            if hasattr(module, 'channel_attention') and hasattr(module, 'spatial_attention'):
                hook = module.register_forward_hook(self._hook_fn)
                self.hooks.append((name, hook))
    
    def _hook_fn(self, module, input, output):
        """
        钩子函数，用于获取注意力图
        """
        # 获取模块名称
        for name, hook in self.hooks:
            if hook.id == id(module._forward_hooks[list(module._forward_hooks.keys())[0]]):
                module_name = name
                break
        else:
            return
        
        # 保存输入特征
        x = input[0].detach()
        
        # 计算通道注意力
        channel_attention = module.channel_attention(x)
        
        # 计算空间注意力
        # 先应用通道注意力
        x_channel = x * channel_attention.expand_as(x)
        # 再计算空间注意力
        spatial_attention = module.spatial_attention(x_channel)
        
        # 保存注意力图
        self.attention_maps[module_name] = {
            'input': x.detach(),
            'channel': channel_attention.detach(),
            'spatial': spatial_attention.detach(),
            'output': output.detach()
        }
    
    def visualize(self, image_tensor, save_dir=None):
        """
        可视化注意力图
        
        参数:
            image_tensor: 输入图像张量 [1, channels, height, width]
            save_dir: 保存目录
        """
        # 前向传播
        with torch.no_grad():
            _ = self.model(image_tensor)
        
        # 如果没有注意力图
        if not self.attention_maps:
            print('未获取到注意力图')
            return
        
        # 创建保存目录
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # 反归一化函数
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        
        # 显示原始图像
        img = inv_normalize(image_tensor[0])
        img = img.permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        
        # 遍历每个CBAM模块
        for i, (module_name, attention) in enumerate(self.attention_maps.items()):
            # 创建图形
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # 显示原始图像
            axes[0, 0].imshow(img)
            axes[0, 0].set_title('原始图像')
            axes[0, 0].axis('off')
            
            # 显示通道注意力
            channel_attention = attention['channel'].squeeze().cpu().numpy()
            axes[0, 1].bar(range(len(channel_attention)), channel_attention)
            axes[0, 1].set_title('通道注意力')
            axes[0, 1].set_xlabel('通道')
            axes[0, 1].set_ylabel('注意力权重')
            
            # 显示空间注意力
            spatial_attention = attention['spatial'].squeeze().cpu().numpy()
            im = axes[0, 2].imshow(spatial_attention, cmap='jet')
            axes[0, 2].set_title('空间注意力')
            axes[0, 2].axis('off')
            plt.colorbar(im, ax=axes[0, 2])
            
            # 显示输入特征图（取平均值）
            input_feature = attention['input'].mean(dim=1).squeeze().cpu().numpy()
            axes[1, 0].imshow(input_feature, cmap='viridis')
            axes[1, 0].set_title('输入特征图')
            axes[1, 0].axis('off')
            
            # 显示应用通道注意力后的特征图
            channel_weighted = (attention['input'] * attention['channel'].expand_as(attention['input']))
            channel_weighted = channel_weighted.mean(dim=1).squeeze().cpu().numpy()
            axes[1, 1].imshow(channel_weighted, cmap='viridis')
            axes[1, 1].set_title('通道加权特征图')
            axes[1, 1].axis('off')
            
            # 显示输出特征图
            output_feature = attention['output'].mean(dim=1).squeeze().cpu().numpy()
            axes[1, 2].imshow(output_feature, cmap='viridis')
            axes[1, 2].set_title('输出特征图')
            axes[1, 2].axis('off')
            
            # 设置标题
            fig.suptitle(f'CBAM模块: {module_name}', fontsize=16)
            
            # 调整布局
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            
            # 保存图形
            if save_dir:
                # 将模块名称中的点替换为下划线
                module_name_safe = module_name.replace('.', '_')
                save_path = os.path.join(save_dir, f'attention_{module_name_safe}.png')
                plt.savefig(save_path)
                print(f'已保存注意力图: {save_path}')
            
            plt.close()
    
    def visualize_attention_on_image(self, image_tensor, save_dir=None):
        """
        将注意力图叠加在原始图像上
        
        参数:
            image_tensor: 输入图像张量 [1, channels, height, width]
            save_dir: 保存目录
        """
        # 前向传播
        with torch.no_grad():
            _ = self.model(image_tensor)
        
        # 如果没有注意力图
        if not self.attention_maps:
            print('未获取到注意力图')
            return
        
        # 创建保存目录
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # 反归一化函数
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        
        # 显示原始图像
        img = inv_normalize(image_tensor[0])
        img = img.permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        
        # 遍历每个CBAM模块
        for module_name, attention in self.attention_maps.items():
            # 获取空间注意力图
            spatial_attention = attention['spatial'].squeeze().cpu().numpy()
            
            # 创建图形
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 显示原始图像
            axes[0].imshow(img)
            axes[0].set_title('原始图像')
            axes[0].axis('off')
            
            # 显示空间注意力图
            axes[1].imshow(spatial_attention, cmap='jet')
            axes[1].set_title('空间注意力')
            axes[1].axis('off')
            
            # 将注意力图叠加在原始图像上
            # 调整注意力图大小以匹配原始图像
            attention_resized = np.resize(spatial_attention, (img.shape[0], img.shape[1]))
            # 创建热力图
            cmap = plt.cm.jet
            attention_heatmap = cmap(attention_resized)
            attention_heatmap = attention_heatmap[:, :, :3]  # 去除alpha通道
            # 叠加图像
            alpha = 0.6  # 透明度
            overlay = img * (1 - alpha) + attention_heatmap * alpha
            
            # 显示叠加图像
            axes[2].imshow(overlay)
            axes[2].set_title('注意力叠加')
            axes[2].axis('off')
            
            # 设置标题
            fig.suptitle(f'CBAM模块: {module_name}', fontsize=16)
            
            # 调整布局
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            
            # 保存图形
            if save_dir:
                # 将模块名称中的点替换为下划线
                module_name_safe = module_name.replace('.', '_')
                save_path = os.path.join(save_dir, f'overlay_{module_name_safe}.png')
                plt.savefig(save_path)
                print(f'已保存叠加图: {save_path}')
            
            plt.close()
    
    def close(self):
        """
        移除所有钩子
        """
        for _, hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attention_maps = {}


def preprocess_image(image_path, image_size=64):
    """
    预处理图像
    
    参数:
        image_path: 图像路径
        image_size: 图像大小
    
    返回:
        torch.Tensor: 预处理后的图像张量
    """
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 定义变换
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=AUGMENTATION['normalize']['mean'],
            std=AUGMENTATION['normalize']['std']
        )
    ])
    
    # 应用变换
    image_tensor = transform(image)
    
    # 添加批次维度
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def main():
    # 获取设备
    device = get_device(DEVICE['use_cuda'], DEVICE['cuda_device'])
    
    # 加载模型
    checkpoint_dir = LOGGING['checkpoint_dir']
    model_path = os.path.join(checkpoint_dir, 'cbam_resnet_best.pth')
    
    if not os.path.exists(model_path):
        print(f'未找到模型: {model_path}')
        return
    
    # 创建模型
    model = CBAMResNet18(num_classes=DATASET['num_classes'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 创建结果目录
    results_dir = LOGGING['results_dir']
    attention_dir = os.path.join(results_dir, 'attention_maps')
    os.makedirs(attention_dir, exist_ok=True)
    
    # 创建注意力可视化器
    visualizer = AttentionVisualizer(model, device)
    
    # 获取数据集中的一些图像
    data_dir = DATASET['data_dir']
    
    # 遍历数据集目录，找到一些图像
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_paths.append(os.path.join(root, file))
                if len(image_paths) >= 5:  # 只处理5张图像
                    break
        if len(image_paths) >= 5:
            break
    
    # 处理每张图像
    for i, image_path in enumerate(image_paths):
        print(f'处理图像 {i+1}/{len(image_paths)}: {image_path}')
        
        # 预处理图像
        image_tensor = preprocess_image(image_path, DATASET['image_size'])
        image_tensor = image_tensor.to(device)
        
        # 创建图像特定的保存目录
        image_name = os.path.basename(image_path).split('.')[0]
        image_dir = os.path.join(attention_dir, f'image_{i+1}_{image_name}')
        os.makedirs(image_dir, exist_ok=True)
        
        # 可视化注意力图
        visualizer.visualize(image_tensor, image_dir)
        
        # 可视化注意力叠加图
        visualizer.visualize_attention_on_image(image_tensor, image_dir)
    
    # 关闭可视化器
    visualizer.close()
    
    print('注意力机制可视化完成')


if __name__ == '__main__':
    main()