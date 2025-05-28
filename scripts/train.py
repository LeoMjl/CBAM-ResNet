import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from models.resnet import ResNet18
from models.cbam_resnet import CBAMResNet18
from scripts.data_loader import get_data_loaders


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs=100, save_dir='logs'):
    """
    训练模型并记录训练过程
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 训练设备
        num_epochs: 训练轮数
        save_dir: 保存模型和日志的目录
        
    返回:
        dict: 包含训练历史的字典
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 初始化训练历史记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 记录最佳验证准确率
    best_val_acc = 0.0
    
    # 使用trange创建带进度条的epoch迭代器
    epoch_iterator = trange(num_epochs, desc="训练进度", unit="epoch")
    
    # 训练循环
    for epoch in epoch_iterator:
        epoch_iterator.set_description(f'Epoch {epoch+1}/{num_epochs}')
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_corrects = 0
        
        # 记录训练开始时间
        start_time = time.time()
        
        # 使用tqdm包装训练数据加载器
        train_bar = tqdm(train_loader, desc="训练批次", leave=False)
        for inputs, labels in train_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
            
            # 统计
            batch_loss = loss.item() * inputs.size(0)
            batch_corrects = torch.sum(preds == labels.data)
            train_loss += batch_loss
            train_corrects += batch_corrects
            
            # 更新进度条信息
            train_bar.set_postfix({
                'loss': batch_loss / inputs.size(0),
                'acc': batch_corrects.double() / inputs.size(0)
            })
        
        # 计算训练集上的损失和准确率
        train_loss = train_loss / len(train_loader.sampler)
        train_acc = train_corrects.double() / len(train_loader.sampler)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        # 使用tqdm包装验证数据加载器
        val_bar = tqdm(test_loader, desc="验证批次", leave=False)
        for inputs, labels in val_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            # 统计
            batch_loss = loss.item() * inputs.size(0)
            batch_corrects = torch.sum(preds == labels.data)
            val_loss += batch_loss
            val_corrects += batch_corrects
            
            # 更新进度条信息
            val_bar.set_postfix({
                'loss': batch_loss / inputs.size(0),
                'acc': batch_corrects.double() / inputs.size(0)
            })
        
        # 计算验证集上的损失和准确率
        val_loss = val_loss / len(test_loader.sampler)
        val_acc = val_corrects.double() / len(test_loader.sampler)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录训练历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.item())
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())
        
        # 计算训练时间
        time_elapsed = time.time() - start_time
        
        # 更新epoch进度条信息
        epoch_iterator.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'train_acc': f'{train_acc:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_acc': f'{val_acc:.4f}',
            'time': f'{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s'
        })
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f'Best model saved with accuracy: {best_val_acc:.4f}')
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    print(f'Training complete. Best val Acc: {best_val_acc:.4f}')
    
    return history


def evaluate_model(model, test_loader, criterion, device):
    """
    评估模型性能
    
    参数:
        model: 要评估的模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 评估设备
        
    返回:
        tuple: (损失, 准确率, 预测标签, 真实标签)
    """
    model.eval()
    test_loss = 0.0
    test_corrects = 0
    all_preds = []
    all_labels = []
    
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # 前向传播
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
        
        # 统计
        test_loss += loss.item() * inputs.size(0)
        test_corrects += torch.sum(preds == labels.data)
        
        # 收集预测和真实标签
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # 计算测试集上的损失和准确率
    test_loss = test_loss / len(test_loader.sampler)
    test_acc = test_corrects.double() / len(test_loader.sampler)
    
    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
    
    return test_loss, test_acc.item(), np.array(all_preds), np.array(all_labels)


def plot_training_history(history, save_path='results/training_history.png'):
    """
    绘制训练历史曲线
    
    参数:
        history: 包含训练历史的字典
        save_path: 保存图像的路径
    """
    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 创建图像
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制准确率曲线
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def compare_models(histories, model_names, save_path='results/model_comparison.png'):
    """
    比较不同模型的性能
    
    参数:
        histories: 包含多个模型训练历史的列表
        model_names: 模型名称列表
        save_path: 保存图像的路径
    """
    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 创建图像
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    for i, history in enumerate(histories):
        ax1.plot(history['val_loss'], label=f'{model_names[i]} Loss')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Validation Loss Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制准确率曲线
    for i, history in enumerate(histories):
        ax2.plot(history['val_acc'], label=f'{model_names[i]} Accuracy')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy Comparison')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()