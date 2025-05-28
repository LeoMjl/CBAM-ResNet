import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils import generate_model_comparison_report, plot_model_comparison
from configs.config import LOGGING


def load_training_history(history_path):
    """
    加载训练历史记录
    
    参数:
        history_path: 历史记录文件路径
    
    返回:
        dict: 训练历史记录
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    return history


def load_classification_report(report_path):
    """
    加载分类报告
    
    参数:
        report_path: 分类报告文件路径
    
    返回:
        dict: 分类报告
    """
    # 读取分类报告文本
    with open(report_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 解析分类报告
    report = {}
    in_metrics = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith('=') or line.startswith('-'):
            continue
        
        if line.startswith('分类报告') or line.startswith('生成时间'):
            continue
        
        if 'precision' in line and 'recall' in line and 'f1-score' in line:
            in_metrics = True
            continue
        
        if in_metrics:
            parts = line.split()
            if len(parts) >= 4:
                # 处理类别名称可能包含空格的情况
                if parts[0] == 'accuracy':
                    report['accuracy'] = float(parts[1])
                elif parts[0] == 'macro' and parts[1] == 'avg':
                    report['macro avg'] = {
                        'precision': float(parts[2]),
                        'recall': float(parts[3]),
                        'f1-score': float(parts[4])
                    }
                elif parts[0] == 'weighted' and parts[1] == 'avg':
                    report['weighted avg'] = {
                        'precision': float(parts[2]),
                        'recall': float(parts[3]),
                        'f1-score': float(parts[4])
                    }
                else:
                    # 处理类别名称
                    class_name = parts[0]
                    i = 1
                    while i < len(parts) - 3 and not parts[i].replace('.', '', 1).isdigit():
                        class_name += ' ' + parts[i]
                        i += 1
                    
                    # 提取指标
                    precision = float(parts[i])
                    recall = float(parts[i+1])
                    f1_score = float(parts[i+2])
                    
                    report[class_name] = {
                        'precision': precision,
                        'recall': recall,
                        'f1-score': f1_score
                    }
    
    return report


def generate_report():
    """
    生成模型对比报告
    """
    # 设置路径
    results_dir = LOGGING['results_dir']
    log_dir = LOGGING['log_dir']
    
    # 加载训练历史记录
    resnet_history_path = os.path.join(log_dir, 'resnet_history.json')
    cbam_resnet_history_path = os.path.join(log_dir, 'cbam_resnet_history.json')
    
    if not os.path.exists(resnet_history_path) or not os.path.exists(cbam_resnet_history_path):
        print('未找到训练历史记录文件')
        return
    
    resnet_history = load_training_history(resnet_history_path)
    cbam_resnet_history = load_training_history(cbam_resnet_history_path)
    
    # 加载分类报告
    resnet_report_path = os.path.join(results_dir, 'resnet_classification_report.txt')
    cbam_resnet_report_path = os.path.join(results_dir, 'cbam_resnet_classification_report.txt')
    
    if not os.path.exists(resnet_report_path) or not os.path.exists(cbam_resnet_report_path):
        print('未找到分类报告文件')
        return
    
    resnet_report = load_classification_report(resnet_report_path)
    cbam_resnet_report = load_classification_report(cbam_resnet_report_path)
    
    # 生成模型对比报告
    report_path = os.path.join(results_dir, 'model_comparison_report.txt')
    generate_model_comparison_report(
        resnet_history, cbam_resnet_history,
        resnet_report, cbam_resnet_report,
        save_path=report_path
    )
    
    # 绘制模型对比图
    comparison_path = os.path.join(results_dir, 'model_comparison.png')
    plot_model_comparison(resnet_history, cbam_resnet_history, save_path=comparison_path)
    
    # 绘制单独的训练历史曲线
    plot_training_history(resnet_history, os.path.join(results_dir, 'resnet_history.png'), 'ResNet')
    plot_training_history(cbam_resnet_history, os.path.join(results_dir, 'cbam_resnet_history.png'), 'CBAM-ResNet')
    
    print('已生成模型对比报告和可视化结果')


def plot_training_history(history, save_path, model_name):
    """
    绘制训练历史曲线
    
    参数:
        history: 训练历史记录
        save_path: 保存路径
        model_name: 模型名称
    """
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    axes[0].plot(history['train_loss'], label='训练损失')
    axes[0].plot(history['val_loss'], label='验证损失')
    axes[0].set_xlabel('轮次')
    axes[0].set_ylabel('损失')
    axes[0].set_title(f'{model_name} 损失曲线')
    axes[0].legend()
    axes[0].grid(True)
    
    # 绘制准确率曲线
    axes[1].plot(history['train_acc'], label='训练准确率')
    axes[1].plot(history['val_acc'], label='验证准确率')
    axes[1].set_xlabel('轮次')
    axes[1].set_ylabel('准确率')
    axes[1].set_title(f'{model_name} 准确率曲线')
    axes[1].legend()
    axes[1].grid(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    plt.savefig(save_path)
    print(f'已保存训练历史曲线: {save_path}')
    
    plt.close()


def plot_class_performance_comparison(resnet_report, cbam_resnet_report, save_path):
    """
    绘制类别性能对比图
    
    参数:
        resnet_report: ResNet模型分类报告
        cbam_resnet_report: CBAM-ResNet模型分类报告
        save_path: 保存路径
    """
    # 提取类别性能数据
    classes = []
    resnet_f1 = []
    cbam_resnet_f1 = []
    
    for class_name, metrics in resnet_report.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            classes.append(class_name)
            resnet_f1.append(metrics['f1-score'])
            cbam_resnet_f1.append(cbam_resnet_report[class_name]['f1-score'])
    
    # 创建DataFrame
    df = pd.DataFrame({
        'Class': classes,
        'ResNet': resnet_f1,
        'CBAM-ResNet': cbam_resnet_f1
    })
    
    # 计算性能差异
    df['Difference'] = df['CBAM-ResNet'] - df['ResNet']
    df['Improvement'] = (df['Difference'] / df['ResNet']) * 100
    
    # 按性能提升排序
    df = df.sort_values('Improvement', ascending=False)
    
    # 选择前10个和后10个类别
    if len(df) > 20:
        top_bottom = pd.concat([df.head(10), df.tail(10)])
    else:
        top_bottom = df
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制条形图
    sns.barplot(x='Improvement', y='Class', data=top_bottom, palette='coolwarm')
    
    # 添加标题和标签
    plt.title('CBAM-ResNet相对于ResNet的性能提升(%)')
    plt.xlabel('性能提升百分比')
    plt.ylabel('类别')
    plt.axvline(x=0, color='black', linestyle='--')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    plt.savefig(save_path)
    print(f'已保存类别性能对比图: {save_path}')
    
    plt.close()


def main():
    # 创建结果目录
    results_dir = LOGGING['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    # 生成报告
    generate_report()
    
    # 加载分类报告
    resnet_report_path = os.path.join(results_dir, 'resnet_classification_report.txt')
    cbam_resnet_report_path = os.path.join(results_dir, 'cbam_resnet_classification_report.txt')
    
    if os.path.exists(resnet_report_path) and os.path.exists(cbam_resnet_report_path):
        resnet_report = load_classification_report(resnet_report_path)
        cbam_resnet_report = load_classification_report(cbam_resnet_report_path)
        
        # 绘制类别性能对比图
        class_performance_path = os.path.join(results_dir, 'class_performance_comparison.png')
        plot_class_performance_comparison(resnet_report, cbam_resnet_report, class_performance_path)


if __name__ == '__main__':
    main()