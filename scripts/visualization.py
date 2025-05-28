import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms
import os
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def visualize_batch(images, labels, class_names, num_images=16, save_path=None):
    """
    可视化一批图像
    
    参数:
        images: 图像张量 [batch_size, channels, height, width]
        labels: 标签张量 [batch_size]
        class_names: 类别名称列表
        num_images: 要显示的图像数量
        save_path: 保存路径
    """
    # 确保不超过批次大小
    batch_size = images.size(0)
    num_images = min(batch_size, num_images)
    
    # 创建网格
    rows = int(np.sqrt(num_images))
    cols = int(np.ceil(num_images / rows))
    
    # 创建图形
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # 反归一化函数
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    # 显示图像
    for i in range(num_images):
        # 反归一化
        img = inv_normalize(images[i])
        # 转换为numpy数组并调整通道顺序
        img = img.permute(1, 2, 0).cpu().numpy()
        # 裁剪到[0, 1]范围
        img = np.clip(img, 0, 1)
        
        # 显示图像
        axes[i].imshow(img)
        axes[i].set_title(class_names[labels[i]])
        axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    if save_path:
        plt.savefig(save_path)
        print(f'已保存批次可视化: {save_path}')
    
    plt.close()


def visualize_model_predictions(model, dataloader, class_names, device, num_images=16, save_path=None):
    """
    可视化模型预测结果
    
    参数:
        model: 模型
        dataloader: 数据加载器
        class_names: 类别名称列表
        device: 计算设备
        num_images: 要显示的图像数量
        save_path: 保存路径
    """
    # 设置模型为评估模式
    model.eval()
    
    # 获取一批数据
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)
    
    # 确保不超过批次大小
    batch_size = images.size(0)
    num_images = min(batch_size, num_images)
    
    # 获取预测结果
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    # 创建网格
    rows = int(np.sqrt(num_images))
    cols = int(np.ceil(num_images / rows))
    
    # 创建图形
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # 反归一化函数
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    # 显示图像和预测结果
    for i in range(num_images):
        # 反归一化
        img = inv_normalize(images[i])
        # 转换为numpy数组并调整通道顺序
        img = img.permute(1, 2, 0).cpu().numpy()
        # 裁剪到[0, 1]范围
        img = np.clip(img, 0, 1)
        
        # 获取真实标签和预测标签
        true_label = class_names[labels[i]]
        pred_label = class_names[preds[i]]
        
        # 设置标题颜色（正确为绿色，错误为红色）
        title_color = 'green' if labels[i] == preds[i] else 'red'
        
        # 显示图像
        axes[i].imshow(img)
        axes[i].set_title(f'真实: {true_label}\n预测: {pred_label}', color=title_color)
        axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    if save_path:
        plt.savefig(save_path)
        print(f'已保存预测可视化: {save_path}')
    
    plt.close()


def visualize_feature_maps(model, image, layer_name, num_features=16, save_path=None):
    """
    可视化模型特征图
    
    参数:
        model: 模型
        image: 输入图像 [1, channels, height, width]
        layer_name: 要可视化的层名称
        num_features: 要显示的特征图数量
        save_path: 保存路径
    """
    # 设置模型为评估模式
    model.eval()
    
    # 注册钩子函数
    features = {}
    def hook_fn(module, input, output):
        features['output'] = output.detach()
    
    # 查找指定层
    for name, module in model.named_modules():
        if name == layer_name:
            hook = module.register_forward_hook(hook_fn)
            break
    else:
        print(f'未找到层: {layer_name}')
        return
    
    # 前向传播
    with torch.no_grad():
        _ = model(image)
    
    # 移除钩子
    hook.remove()
    
    # 获取特征图
    feature_maps = features['output'][0]  # [channels, height, width]
    num_features = min(feature_maps.size(0), num_features)
    
    # 创建网格
    rows = int(np.sqrt(num_features))
    cols = int(np.ceil(num_features / rows))
    
    # 创建图形
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # 显示特征图
    for i in range(num_features):
        # 获取特征图
        feature_map = feature_maps[i].cpu().numpy()
        
        # 显示特征图
        im = axes[i].imshow(feature_map, cmap='viridis')
        axes[i].set_title(f'特征 {i+1}')
        axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(num_features, len(axes)):
        axes[i].axis('off')
    
    # 添加颜色条
    fig.colorbar(im, ax=axes.ravel().tolist())
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    if save_path:
        plt.savefig(save_path)
        print(f'已保存特征图可视化: {save_path}')
    
    plt.close()


def visualize_attention_maps(model, image, save_path=None):
    """
    可视化CBAM注意力图
    
    参数:
        model: 带CBAM的模型
        image: 输入图像 [1, channels, height, width]
        save_path: 保存路径
    """
    # 设置模型为评估模式
    model.eval()
    
    # 注册钩子函数
    attention_maps = {}
    def hook_fn(module, input, output):
        # 对于CBAM模块，我们需要获取通道注意力和空间注意力
        if hasattr(module, 'channel_attention') and hasattr(module, 'spatial_attention'):
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
            attention_maps[module._get_name()] = {
                'channel': channel_attention.detach(),
                'spatial': spatial_attention.detach()
            }
    
    # 查找所有CBAM模块
    hooks = []
    for name, module in model.named_modules():
        if 'cbam' in name.lower() and hasattr(module, 'channel_attention') and hasattr(module, 'spatial_attention'):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    
    # 如果没有找到CBAM模块
    if not hooks:
        print('未找到CBAM模块')
        return
    
    # 前向传播
    with torch.no_grad():
        _ = model(image)
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    # 如果没有注意力图
    if not attention_maps:
        print('未获取到注意力图')
        return
    
    # 创建图形
    num_modules = len(attention_maps)
    fig, axes = plt.subplots(num_modules, 3, figsize=(12, num_modules * 4))
    
    # 如果只有一个模块，确保axes是二维的
    if num_modules == 1:
        axes = np.expand_dims(axes, axis=0)
    
    # 反归一化函数
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    # 显示原始图像
    img = inv_normalize(image[0])
    img = img.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    
    # 遍历每个CBAM模块
    for i, (module_name, attention) in enumerate(attention_maps.items()):
        # 显示原始图像
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('原始图像')
        axes[i, 0].axis('off')
        
        # 显示通道注意力
        # 通道注意力是一个一维向量，我们将其可视化为条形图
        channel_attention = attention['channel'].squeeze().cpu().numpy()
        axes[i, 1].bar(range(len(channel_attention)), channel_attention)
        axes[i, 1].set_title(f'{module_name}\n通道注意力')
        axes[i, 1].set_xlabel('通道')
        axes[i, 1].set_ylabel('注意力权重')
        
        # 显示空间注意力
        spatial_attention = attention['spatial'].squeeze().cpu().numpy()
        im = axes[i, 2].imshow(spatial_attention, cmap='jet')
        axes[i, 2].set_title(f'{module_name}\n空间注意力')
        axes[i, 2].axis('off')
        plt.colorbar(im, ax=axes[i, 2])
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    if save_path:
        plt.savefig(save_path)
        print(f'已保存注意力图可视化: {save_path}')
    
    plt.close()


def visualize_features_tsne(model, dataloader, class_names, device, num_samples=500, save_path=None):
    """
    使用t-SNE可视化模型特征
    
    参数:
        model: 模型
        dataloader: 数据加载器
        class_names: 类别名称列表
        device: 计算设备
        num_samples: 要使用的样本数量
        save_path: 保存路径
    """
    # 设置模型为评估模式
    model.eval()
    
    # 注册钩子函数
    features = []
    labels_list = []
    
    def hook_fn(module, input, output):
        # 我们只关心全连接层之前的特征
        features.append(input[0].detach().cpu())
    
    # 查找最后一个全连接层
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Linear):
            hook = module.register_forward_hook(hook_fn)
            break
    else:
        print('未找到全连接层')
        return
    
    # 收集特征和标签
    with torch.no_grad():
        for images, labels in dataloader:
            if len(features) * images.size(0) >= num_samples:
                break
            
            images = images.to(device)
            _ = model(images)
            labels_list.append(labels)
    
    # 移除钩子
    hook.remove()
    
    # 处理收集到的特征和标签
    features = torch.cat(features, dim=0)[:num_samples]
    labels = torch.cat(labels_list, dim=0)[:num_samples]
    
    # 将特征展平
    features = features.view(features.size(0), -1).numpy()
    labels = labels.numpy()
    
    # 使用PCA进行降维（如果特征维度很高）
    if features.shape[1] > 50:
        pca = PCA(n_components=50)
        features = pca.fit_transform(features)
    
    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features)
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 设置颜色映射
    num_classes = len(class_names)
    cmap = plt.cm.get_cmap('tab20', num_classes)
    
    # 绘制散点图
    for i in range(num_classes):
        idx = labels == i
        if np.any(idx):
            plt.scatter(features_tsne[idx, 0], features_tsne[idx, 1], c=[cmap(i)], label=class_names[i], alpha=0.7)
    
    # 添加图例
    if num_classes <= 10:
        plt.legend()
    else:
        # 如果类别太多，只显示部分图例
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[:10], labels[:10], loc='upper right', title='前10个类别')
    
    plt.title('t-SNE特征可视化')
    plt.xlabel('t-SNE特征1')
    plt.ylabel('t-SNE特征2')
    plt.grid(True)
    
    # 保存图形
    if save_path:
        plt.savefig(save_path)
        print(f'已保存t-SNE特征可视化: {save_path}')
    
    plt.close()