import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet import ResNet18
from models.cbam_resnet import CBAMResNet18
from scripts.data_loader import get_data_loaders
from scripts.utils import get_device, plot_confusion_matrix, save_classification_report
from scripts.visualization import visualize_model_predictions, visualize_attention_maps, visualize_features_tsne
from configs.config import DATASET, DATALOADER, MODEL, DEVICE, LOGGING


def load_model(model_path, model_type, num_classes, device):
    """
    加载训练好的模型
    
    参数:
        model_path: 模型路径
        model_type: 模型类型 ('resnet' 或 'cbam_resnet')
        num_classes: 类别数量
        device: 计算设备
    
    返回:
        model: 加载的模型
    """
    # 创建模型
    if model_type == 'resnet':
        model = ResNet18(num_classes=num_classes, dropout_rate=MODEL['resnet']['dropout_rate'])
    elif model_type == 'cbam_resnet':
        model = CBAMResNet18(num_classes=num_classes, dropout_rate=MODEL['cbam_resnet']['dropout_rate'])
    else:
        raise ValueError(f'不支持的模型类型: {model_type}')
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f'已加载模型: {model_path}')
    return model


def evaluate_model(model, test_loader, device, class_names, results_dir, model_name):
    """
    评估模型性能
    
    参数:
        model: 模型
        test_loader: 测试数据加载器
        device: 计算设备
        class_names: 类别名称列表
        results_dir: 结果保存目录
        model_name: 模型名称
    
    返回:
        dict: 评估结果
    """
    # 设置模型为评估模式
    model.eval()
    
    # 初始化变量
    all_preds = []
    all_labels = []
    test_loss = 0.0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    # 在测试集上评估模型
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 计算损失和准确率
            test_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # 收集预测结果和标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算平均损失和准确率
    test_loss = test_loss / total
    test_acc = correct / total
    
    # 计算精确率、召回率和F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted')
    
    # 打印评估结果
    print(f'测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}')
    print(f'精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}')
    
    # 绘制混淆矩阵
    cm_path = os.path.join(results_dir, f'{model_name}_confusion_matrix.png')
    plot_confusion_matrix(all_labels, all_preds, class_names, save_path=cm_path)
    
    # 保存分类报告
    report_path = os.path.join(results_dir, f'{model_name}_classification_report.txt')
    report = save_classification_report(all_labels, all_preds, class_names, save_path=report_path)
    
    # 可视化模型预测
    pred_path = os.path.join(results_dir, f'{model_name}_predictions.png')
    visualize_model_predictions(model, test_loader, class_names, device, save_path=pred_path)
    
    # 如果是CBAM-ResNet模型，可视化注意力图
    if 'cbam' in model_name.lower():
        # 获取一个样本图像
        images, _ = next(iter(test_loader))
        image = images[0:1].to(device)
        
        # 可视化注意力图
        attn_path = os.path.join(results_dir, f'{model_name}_attention_maps.png')
        visualize_attention_maps(model, image, save_path=attn_path)
    
    # 可视化特征分布
    tsne_path = os.path.join(results_dir, f'{model_name}_tsne.png')
    visualize_features_tsne(model, test_loader, class_names, device, save_path=tsne_path)
    
    # 返回评估结果
    return {
        'loss': test_loss,
        'accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix_path': cm_path,
        'classification_report_path': report_path,
        'predictions_path': pred_path,
        'tsne_path': tsne_path
    }


def main():
    # 创建结果目录
    results_dir = LOGGING['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    # 获取设备
    device = get_device(DEVICE['use_cuda'], DEVICE['cuda_device'])
    
    # 加载数据
    _, test_loader, class_names = get_data_loaders(
        data_dir=DATASET['data_dir'],
        people_csv=DATASET['people_csv'],
        num_classes=DATASET['num_classes'],
        min_images_per_class=DATASET['min_images_per_class'],
        image_size=DATASET['image_size'],
        batch_size=DATALOADER['batch_size'],
        num_workers=DATALOADER['num_workers'],
        train_ratio=DATASET['train_ratio'],
        random_seed=DATASET['random_seed']
    )
    
    # 加载ResNet模型
    resnet_model_path = os.path.join(LOGGING['checkpoint_dir'], 'resnet_best.pth')
    if os.path.exists(resnet_model_path):
        resnet_model = load_model(resnet_model_path, 'resnet', len(class_names), device)
        resnet_results = evaluate_model(resnet_model, test_loader, device, class_names, results_dir, 'resnet')
    else:
        print(f'未找到ResNet模型: {resnet_model_path}')
        resnet_results = None
    
    # 加载CBAM-ResNet模型
    cbam_model_path = os.path.join(LOGGING['checkpoint_dir'], 'cbam_resnet_best.pth')
    if os.path.exists(cbam_model_path):
        cbam_model = load_model(cbam_model_path, 'cbam_resnet', len(class_names), device)
        cbam_results = evaluate_model(cbam_model, test_loader, device, class_names, results_dir, 'cbam_resnet')
    else:
        print(f'未找到CBAM-ResNet模型: {cbam_model_path}')
        cbam_results = None
    
    # 比较模型性能
    if resnet_results and cbam_results:
        print('\n模型性能对比:')
        print(f"{'指标':<15} {'ResNet':<15} {'CBAM-ResNet':<15} {'差异':<10} {'提升百分比':<15}")
        print('-' * 70)
        
        # 比较准确率
        acc_diff = cbam_results['accuracy'] - resnet_results['accuracy']
        acc_improvement = (acc_diff / resnet_results['accuracy']) * 100
        print(f"{'准确率':<15} {resnet_results['accuracy']:.4f}{'':>9} {cbam_results['accuracy']:.4f}{'':>9} {acc_diff:.4f}{'':>4} {acc_improvement:.2f}%{'':>8}")
        
        # 比较F1分数
        f1_diff = cbam_results['f1'] - resnet_results['f1']
        f1_improvement = (f1_diff / resnet_results['f1']) * 100
        print(f"{'F1分数':<15} {resnet_results['f1']:.4f}{'':>9} {cbam_results['f1']:.4f}{'':>9} {f1_diff:.4f}{'':>4} {f1_improvement:.2f}%{'':>8}")
        
        # 输出结论
        print('\n结论:')
        if acc_diff > 0:
            print(f"CBAM注意力机制提高了模型的准确率，提升了{acc_improvement:.2f}%。")
        else:
            print(f"CBAM注意力机制未能提高模型的准确率，降低了{abs(acc_improvement):.2f}%。")
        
        if f1_diff > 0:
            print(f"CBAM注意力机制提高了模型的F1分数，提升了{f1_improvement:.2f}%。")
        else:
            print(f"CBAM注意力机制未能提高模型的F1分数，降低了{abs(f1_improvement):.2f}%。")


if __name__ == '__main__':
    main()