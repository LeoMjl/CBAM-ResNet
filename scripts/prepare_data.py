import os
import zipfile
import shutil
import pandas as pd


def extract_zip(zip_path, extract_to):
    """
    解压ZIP文件
    
    参数:
        zip_path: ZIP文件路径
        extract_to: 解压目标路径
    """
    # 确保目标目录存在
    os.makedirs(extract_to, exist_ok=True)
    
    # 解压文件
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f'已将 {zip_path} 解压到 {extract_to}')


def create_people_csv(lfw_dir, output_file):
    """
    创建包含人名和图像数量的CSV文件
    
    参数:
        lfw_dir: LFW数据集目录
        output_file: 输出CSV文件路径
    """
    # 获取所有人名目录
    people = []
    for person_dir in os.listdir(lfw_dir):
        person_path = os.path.join(lfw_dir, person_dir)
        if os.path.isdir(person_path):
            # 计算图像数量
            image_count = len([f for f in os.listdir(person_path) if f.endswith('.jpg')])
            people.append({'name': person_dir, 'images': image_count})
    
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(people)
    df.to_csv(output_file, index=False)
    
    print(f'已创建人物信息CSV文件: {output_file}')
    print(f'共有 {len(people)} 个人物类别')


def main():
    # 设置路径
    data_dir = 'data'
    zip_path = os.path.join(data_dir, 'archive.zip')
    extract_to = data_dir
    lfw_dir = os.path.join(data_dir, 'lfw-deepfunneled', 'lfw-deepfunneled')
    output_file = os.path.join(data_dir, 'people.csv')
    
    # 检查ZIP文件是否存在
    if not os.path.exists(zip_path):
        print(f'错误: 未找到ZIP文件 {zip_path}')
        return
    
    # 解压ZIP文件（如果需要）
    if not os.path.exists(lfw_dir):
        print('解压数据集...')
        extract_zip(zip_path, extract_to)
    else:
        print(f'数据集已存在于 {lfw_dir}')
    
    # 创建人物信息CSV文件（如果需要）
    if not os.path.exists(output_file):
        print('创建人物信息CSV文件...')
        create_people_csv(lfw_dir, output_file)
    else:
        print(f'人物信息CSV文件已存在: {output_file}')
    
    print('数据准备完成!')


if __name__ == '__main__':
    main()