import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import time
from datetime import datetime


def set_seed(seed):
    """
    设置随机种子以确保结果可复现
    
    参数:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f'已设置随机种子: {seed}')


def get_device(use_cuda=True, cuda_device=0):
    """
    获取计算设备
    
    参数:
        use_cuda: 是否使用CUDA
        cuda_device: CUDA设备ID
    
    返回:
        torch.device: 计算设备
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{cuda_device}')
        print(f'使用GPU: {torch.cuda.get_device_name(cuda_device)}')
    else:
        device = torch.device('cpu')
        print('使用CPU')
    return device


def create_directories(directories):
    """
    创建目录（如果不存在）
    
    参数:
        directories: 目录路径列表
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f'已创建目录: {directory}')


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    绘制混淆矩阵
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        save_path: 保存路径
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 创建图形
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    if save_path:
        plt.savefig(save_path)
        print(f'已保存混淆矩阵: {save_path}')
    
    plt.close()


def save_classification_report(y_true, y_pred, class_names, save_path=None):
    """
    保存分类报告
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        save_path: 保存路径
    
    返回:
        dict: 分类报告字典
    """
    # 生成分类报告
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # 转换为DataFrame以便于显示和保存
    df_report = pd.DataFrame(report).transpose()
    
    # 保存报告
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('分类报告\n')
            f.write('=' * 80 + '\n')
            f.write(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
            f.write(classification_report(y_true, y_pred, target_names=class_names))
        print(f'已保存分类报告: {save_path}')
    
    return report


def generate_model_comparison_report(resnet_history, cbam_resnet_history, resnet_report, cbam_resnet_report, save_path=None):
    """
    生成模型对比报告
    
    参数:
        resnet_history: ResNet模型训练历史
        cbam_resnet_history: CBAM-ResNet模型训练历史
        resnet_report: ResNet模型分类报告
        cbam_resnet_report: CBAM-ResNet模型分类报告
        save_path: 保存路径
    """
    # 提取性能指标
    resnet_acc = resnet_report['accuracy']
    cbam_resnet_acc = cbam_resnet_report['accuracy']
    
    resnet_macro_f1 = resnet_report['macro avg']['f1-score']
    cbam_resnet_macro_f1 = cbam_resnet_report['macro avg']['f1-score']
    
    resnet_weighted_f1 = resnet_report['weighted avg']['f1-score']
    cbam_resnet_weighted_f1 = cbam_resnet_report['weighted avg']['f1-score']
    
    # 计算训练时间和收敛速度
    resnet_train_time = sum(resnet_history['epoch_time'])
    cbam_resnet_train_time = sum(cbam_resnet_history['epoch_time'])
    
    # 找到达到最佳验证准确率的轮次
    resnet_best_epoch = np.argmax(resnet_history['val_acc'])
    cbam_resnet_best_epoch = np.argmax(cbam_resnet_history['val_acc'])
    
    # 生成报告内容
    report_content = "模型对比报告\n"
    report_content += "=" * 80 + "\n"
    report_content += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    report_content += "1. 性能指标对比\n"
    report_content += "-" * 40 + "\n"
    report_content += f"{'指标':<20} {'ResNet':<15} {'CBAM-ResNet':<15} {'差异':<10} {'提升百分比':<15}\n"
    report_content += "-" * 80 + "\n"
    
    # 计算准确率差异和提升百分比
    acc_diff = cbam_resnet_acc - resnet_acc
    acc_improvement = (acc_diff / resnet_acc) * 100 if resnet_acc > 0 else 0
    report_content += f"{'准确率':<20} {resnet_acc:.4f}{'':>9} {cbam_resnet_acc:.4f}{'':>9} {acc_diff:.4f}{'':>4} {acc_improvement:.2f}%{'':>8}\n"
    
    # 计算宏平均F1分数差异和提升百分比
    macro_f1_diff = cbam_resnet_macro_f1 - resnet_macro_f1
    macro_f1_improvement = (macro_f1_diff / resnet_macro_f1) * 100 if resnet_macro_f1 > 0 else 0
    report_content += f"{'宏平均F1分数':<20} {resnet_macro_f1:.4f}{'':>9} {cbam_resnet_macro_f1:.4f}{'':>9} {macro_f1_diff:.4f}{'':>4} {macro_f1_improvement:.2f}%{'':>8}\n"
    
    # 计算加权平均F1分数差异和提升百分比
    weighted_f1_diff = cbam_resnet_weighted_f1 - resnet_weighted_f1
    weighted_f1_improvement = (weighted_f1_diff / resnet_weighted_f1) * 100 if resnet_weighted_f1 > 0 else 0
    report_content += f"{'加权平均F1分数':<20} {resnet_weighted_f1:.4f}{'':>9} {cbam_resnet_weighted_f1:.4f}{'':>9} {weighted_f1_diff:.4f}{'':>4} {weighted_f1_improvement:.2f}%{'':>8}\n"
    
    report_content += "\n2. 训练效率对比\n"
    report_content += "-" * 40 + "\n"
    report_content += f"{'指标':<20} {'ResNet':<15} {'CBAM-ResNet':<15} {'差异':<10}\n"
    report_content += "-" * 80 + "\n"
    
    # 计算训练时间差异
    train_time_diff = cbam_resnet_train_time - resnet_train_time
    report_content += f"{'总训练时间(秒)':<20} {resnet_train_time:.2f}{'':>9} {cbam_resnet_train_time:.2f}{'':>9} {train_time_diff:.2f}{'':>4}\n"
    
    # 计算收敛速度差异
    convergence_diff = cbam_resnet_best_epoch - resnet_best_epoch
    report_content += f"{'最佳性能轮次':<20} {resnet_best_epoch}{'':>13} {cbam_resnet_best_epoch}{'':>13} {convergence_diff}{'':>8}\n"
    
    report_content += "\n3. 结论\n"
    report_content += "-" * 40 + "\n"
    
    # 根据性能指标生成结论
    if acc_diff > 0:
        report_content += f"CBAM注意力机制提高了模型的准确率，提升了{acc_improvement:.2f}%。\n"
    else:
        report_content += f"CBAM注意力机制未能提高模型的准确率，降低了{abs(acc_improvement):.2f}%。\n"
    
    if macro_f1_diff > 0:
        report_content += f"CBAM注意力机制提高了模型的宏平均F1分数，提升了{macro_f1_improvement:.2f}%。\n"
    else:
        report_content += f"CBAM注意力机制未能提高模型的宏平均F1分数，降低了{abs(macro_f1_improvement):.2f}%。\n"
    
    if train_time_diff > 0:
        report_content += f"添加CBAM注意力机制增加了{train_time_diff:.2f}秒的训练时间。\n"
    else:
        report_content += f"添加CBAM注意力机制减少了{abs(train_time_diff):.2f}秒的训练时间。\n"
    
    if convergence_diff > 0:
        report_content += f"CBAM-ResNet模型需要更多的轮次({convergence_diff}轮)才能达到最佳性能。\n"
    elif convergence_diff < 0:
        report_content += f"CBAM-ResNet模型更快地达到最佳性能，比ResNet模型少用了{abs(convergence_diff)}轮。\n"
    else:
        report_content += "两个模型达到最佳性能所需的轮次相同。\n"
    
    report_content += "\n总体而言，"
    if acc_improvement > 0 and macro_f1_improvement > 0:
        report_content += "CBAM注意力机制显著提高了模型的性能，尽管增加了一些计算开销。"
        report_content += "这表明注意力机制能够帮助模型更好地关注图像中的重要特征，提高识别准确率。"
    elif acc_improvement > 0 or macro_f1_improvement > 0:
        report_content += "CBAM注意力机制在某些方面提高了模型的性能，但效果不是全面的。"
        report_content += "这可能表明对于当前任务，注意力机制的效果有限或需要进一步调整。"
    else:
        report_content += "在当前实验设置下，CBAM注意力机制未能提高模型性能。"
        report_content += "这可能是由于数据集特性、模型配置或超参数选择等因素导致的。"
    
    # 保存报告
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f'已保存模型对比报告: {save_path}')
    
    return report_content


def plot_model_comparison(resnet_history, cbam_resnet_history, save_path=None):
    """
    绘制模型对比图
    
    参数:
        resnet_history: ResNet模型训练历史
        cbam_resnet_history: CBAM-ResNet模型训练历史
        save_path: 保存路径
    """
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 绘制训练损失
    axes[0, 0].plot(resnet_history['train_loss'], label='ResNet')
    axes[0, 0].plot(cbam_resnet_history['train_loss'], label='CBAM-ResNet')
    axes[0, 0].set_xlabel('轮次')
    axes[0, 0].set_ylabel('训练损失')
    axes[0, 0].set_title('训练损失对比')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 绘制验证损失
    axes[0, 1].plot(resnet_history['val_loss'], label='ResNet')
    axes[0, 1].plot(cbam_resnet_history['val_loss'], label='CBAM-ResNet')
    axes[0, 1].set_xlabel('轮次')
    axes[0, 1].set_ylabel('验证损失')
    axes[0, 1].set_title('验证损失对比')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 绘制训练准确率
    axes[1, 0].plot(resnet_history['train_acc'], label='ResNet')
    axes[1, 0].plot(cbam_resnet_history['train_acc'], label='CBAM-ResNet')
    axes[1, 0].set_xlabel('轮次')
    axes[1, 0].set_ylabel('训练准确率')
    axes[1, 0].set_title('训练准确率对比')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 绘制验证准确率
    axes[1, 1].plot(resnet_history['val_acc'], label='ResNet')
    axes[1, 1].plot(cbam_resnet_history['val_acc'], label='CBAM-ResNet')
    axes[1, 1].set_xlabel('轮次')
    axes[1, 1].set_ylabel('验证准确率')
    axes[1, 1].set_title('验证准确率对比')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    if save_path:
        plt.savefig(save_path)
        print(f'已保存模型对比图: {save_path}')
    
    plt.close()


class Timer:
    """
    计时器类，用于测量代码块的执行时间
    """
    def __init__(self, name=None):
        self.name = name
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        if self.name:
            print(f'{self.name} 耗时: {self.interval:.4f} 秒')