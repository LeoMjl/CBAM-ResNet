import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from models.resnet import ResNet18
from models.cbam_resnet import CBAMResNet18
from scripts.data_loader import get_data_loaders, visualize_batch
from scripts.train import train_model, evaluate_model, plot_training_history, compare_models


def main():
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 设置超参数
    batch_size = 16
    image_size = 64
    num_classes = 50
    min_images_per_class = 5
    num_epochs = 100
    learning_rate = 0.001
    weight_decay = 1e-4
    
    # 创建目录
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 加载数据
    data_dir = 'data/lfw-deepfunneled/lfw-deepfunneled'
    csv_file = 'data/people.csv'
    
    print('Loading data...')
    train_loader, test_loader, classes = get_data_loaders(
        data_dir=data_dir,
        csv_file=csv_file,
        batch_size=batch_size,
        image_size=image_size,
        num_classes=num_classes,
        min_images_per_class=min_images_per_class
    )
    print(f'Classes: {classes}')
    print(f'Number of training samples: {len(train_loader.sampler)}')
    print(f'Number of testing samples: {len(test_loader.sampler)}')
    
    # 可视化一个批次的数据
    print('Visualizing a batch of data...')
    visualize_batch(train_loader, classes)
    
    # 创建模型
    print('Creating models...')
    resnet_model = ResNet18(num_classes=num_classes).to(device)
    cbam_resnet_model = CBAMResNet18(num_classes=num_classes).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    resnet_optimizer = optim.Adam(resnet_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    cbam_resnet_optimizer = optim.Adam(cbam_resnet_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    resnet_scheduler = ReduceLROnPlateau(resnet_optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    cbam_resnet_scheduler = ReduceLROnPlateau(cbam_resnet_optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    
    # 训练ResNet模型
    print('\nTraining ResNet model...')
    resnet_history = train_model(
        model=resnet_model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=resnet_optimizer,
        scheduler=resnet_scheduler,
        device=device,
        num_epochs=num_epochs,
        save_dir='logs/resnet'
    )
    
    # 训练带CBAM的ResNet模型
    print('\nTraining CBAM-ResNet model...')
    cbam_resnet_history = train_model(
        model=cbam_resnet_model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=cbam_resnet_optimizer,
        scheduler=cbam_resnet_scheduler,
        device=device,
        num_epochs=num_epochs,
        save_dir='logs/cbam_resnet'
    )
    
    # 绘制训练历史
    print('\nPlotting training history...')
    plot_training_history(resnet_history, save_path='results/resnet_history.png')
    plot_training_history(cbam_resnet_history, save_path='results/cbam_resnet_history.png')
    
    # 比较模型性能
    print('\nComparing models...')
    compare_models(
        histories=[resnet_history, cbam_resnet_history],
        model_names=['ResNet', 'CBAM-ResNet'],
        save_path='results/model_comparison.png'
    )
    
    # 加载最佳模型进行评估
    print('\nEvaluating best models...')
    
    # 加载最佳ResNet模型
    resnet_model.load_state_dict(torch.load('logs/resnet/best_model.pth'))
    resnet_loss, resnet_acc, resnet_preds, resnet_labels = evaluate_model(
        model=resnet_model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )
    
    # 加载最佳CBAM-ResNet模型
    cbam_resnet_model.load_state_dict(torch.load('logs/cbam_resnet/best_model.pth'))
    cbam_resnet_loss, cbam_resnet_acc, cbam_resnet_preds, cbam_resnet_labels = evaluate_model(
        model=cbam_resnet_model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )
    
    # 生成混淆矩阵
    print('\nGenerating confusion matrices...')
    
    # ResNet混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(resnet_labels, resnet_preds)
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.title('ResNet Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('results/resnet_confusion_matrix.png')
    plt.close()
    
    # CBAM-ResNet混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(cbam_resnet_labels, cbam_resnet_preds)
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.title('CBAM-ResNet Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('results/cbam_resnet_confusion_matrix.png')
    plt.close()
    
    # 生成分类报告
    print('\nGenerating classification reports...')
    
    # 获取实际在预测和标签中出现的类别
    unique_labels = np.unique(np.concatenate([resnet_labels, resnet_preds, cbam_resnet_labels, cbam_resnet_preds]))
    actual_classes = [classes[i] for i in unique_labels]
    
    # ResNet分类报告
    resnet_report = classification_report(resnet_labels, resnet_preds, target_names=actual_classes)
    with open('results/resnet_classification_report.txt', 'w') as f:
        f.write(resnet_report)
    print('ResNet Classification Report:')
    print(resnet_report)
    
    # CBAM-ResNet分类报告
    cbam_resnet_report = classification_report(cbam_resnet_labels, cbam_resnet_preds, target_names=actual_classes)
    with open('results/cbam_resnet_classification_report.txt', 'w') as f:
        f.write(cbam_resnet_report)
    print('CBAM-ResNet Classification Report:')
    print(cbam_resnet_report)
    
    # 生成模型对比报告
    print('\nGenerating model comparison report...')
    with open('results/model_comparison_report.txt', 'w') as f:
        f.write('# 模型对比报告\n\n')
        f.write('## 性能指标\n\n')
        f.write(f'| 模型 | 测试损失 | 测试准确率 |\n')
        f.write(f'| --- | --- | --- |\n')
        f.write(f'| ResNet | {resnet_loss:.4f} | {resnet_acc:.4f} |\n')
        f.write(f'| CBAM-ResNet | {cbam_resnet_loss:.4f} | {cbam_resnet_acc:.4f} |\n\n')
        
        f.write('## 结论\n\n')
        if cbam_resnet_acc > resnet_acc:
            f.write('CBAM注意力机制有效提高了ResNet模型的性能，表现在:\n\n')
            f.write(f'1. 测试准确率提高了 {(cbam_resnet_acc - resnet_acc) * 100:.2f}%\n')
            f.write(f'2. 测试损失降低了 {(resnet_loss - cbam_resnet_loss):.4f}\n\n')
            f.write('CBAM注意力机制通过同时关注通道和空间维度的重要特征，增强了模型的表示能力，从而提高了分类性能。')
        elif cbam_resnet_acc < resnet_acc:
            f.write('在本实验中，CBAM注意力机制未能提高ResNet模型的性能，表现在:\n\n')
            f.write(f'1. 测试准确率降低了 {(resnet_acc - cbam_resnet_acc) * 100:.2f}%\n')
            f.write(f'2. 测试损失增加了 {(cbam_resnet_loss - resnet_loss):.4f}\n\n')
            f.write('可能的原因包括:\n\n')
            f.write('1. 数据集规模较小，注意力机制的优势未能充分发挥\n')
            f.write('2. 模型参数设置可能需要进一步优化\n')
            f.write('3. 训练轮数可能不足以让CBAM模型充分收敛')
        else:
            f.write('CBAM注意力机制与原始ResNet模型性能相当，没有显著差异。')
    
    print('Done!')


if __name__ == '__main__':
    main()