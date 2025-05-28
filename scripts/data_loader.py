import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


class LFWDataset(Dataset):
    """
    LFW数据集加载类
    加载LFW数据集中的图像和标签
    """
    def __init__(self, root_dir, csv_file, transform=None, num_classes=50, min_images_per_class=5):
        """
        初始化LFW数据集
        
        参数:
            root_dir (str): 数据集根目录
            csv_file (str): 包含人名和图像数量的CSV文件路径
            transform (callable, optional): 应用于图像的转换
            num_classes (int): 使用的类别数量
            min_images_per_class (int): 每个类别的最小图像数量
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        
        # 筛选出图像数量大于等于min_images_per_class的类别
        filtered_df = df[df['images'] >= min_images_per_class]
        
        # 按图像数量降序排序并取前num_classes个类别
        self.classes = filtered_df.sort_values('images', ascending=False)['name'].tolist()[:num_classes]
        
        # 创建类别到索引的映射
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # 收集所有图像路径和标签
        self.image_paths = []
        self.labels = []
        
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if os.path.isdir(cls_dir):
                for img_name in os.listdir(cls_dir):
                    if img_name.endswith('.jpg'):
                        self.image_paths.append(os.path.join(cls_dir, img_name))
                        self.labels.append(self.class_to_idx[cls])
    
    def __len__(self):
        """
        返回数据集中的样本数量
        """
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        获取指定索引的样本
        
        参数:
            idx (int): 样本索引
            
        返回:
            tuple: (图像, 标签)
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_data_loaders(data_dir, csv_file, batch_size=16, image_size=64, num_classes=50, min_images_per_class=5):
    """
    创建训练和测试数据加载器
    
    参数:
        data_dir (str): 数据集根目录
        csv_file (str): 包含人名和图像数量的CSV文件路径
        batch_size (int): 批量大小
        image_size (int): 图像大小
        num_classes (int): 使用的类别数量
        min_images_per_class (int): 每个类别的最小图像数量
        
    返回:
        tuple: (训练数据加载器, 测试数据加载器, 类别列表)
    """
    # 定义数据转换
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建完整数据集
    full_dataset = LFWDataset(
        root_dir=data_dir,
        csv_file=csv_file,
        transform=None,  # 暂时不应用转换
        num_classes=num_classes,
        min_images_per_class=min_images_per_class
    )
    
    # 获取类别列表
    classes = full_dataset.classes
    
    # 计算训练集和测试集的大小
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    
    # 随机分割数据集
    train_indices, test_indices = torch.utils.data.random_split(
        range(total_size), [train_size, test_size]
    )
    
    # 创建训练集和测试集
    train_dataset = LFWDataset(
        root_dir=data_dir,
        csv_file=csv_file,
        transform=train_transform,
        num_classes=num_classes,
        min_images_per_class=min_images_per_class
    )
    
    test_dataset = LFWDataset(
        root_dir=data_dir,
        csv_file=csv_file,
        transform=test_transform,
        num_classes=num_classes,
        min_images_per_class=min_images_per_class
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(test_indices),
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, test_loader, classes


def visualize_batch(dataloader, classes, num_images=8):
    """
    可视化一个批次的图像
    
    参数:
        dataloader: 数据加载器
        classes: 类别列表
        num_images: 要显示的图像数量
    """
    # 获取一个批次
    images, labels = next(iter(dataloader))
    
    # 创建图像网格
    fig, axes = plt.subplots(2, num_images // 2, figsize=(15, 6))
    axes = axes.flatten()
    
    # 反归一化函数
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return tensor * std + mean
    
    # 显示图像
    for i in range(min(num_images, len(images))):
        img = denormalize(images[i]).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f'Class: {classes[labels[i]]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('data_visualization.png')
    plt.close()