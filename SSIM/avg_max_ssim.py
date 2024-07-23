import os
import cv2
import numpy as np
import torch
from torchvision.transforms import Resize
from pytorch_ssim import ssim as ssim_module
import csv

def load_and_resize_images_from_folder(folder, target_size=(32, 32), device=None):
    images = []
    resize_transform = Resize(target_size)
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(np.rollaxis(img, 2)).float() / 255.0
            resized_img = resize_transform(img_tensor)
            if device is not None:
                resized_img = resized_img.to(device)
            images.append(resized_img)
    return images


def calculate_max_ssim(img_folder1, img_folder2, target_size=(32, 32), window_size=11):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    images1 = load_and_resize_images_from_folder(img_folder1, target_size, device)
    images2 = load_and_resize_images_from_folder(img_folder2, target_size, device)

    max_ssims = []
    output_file = '/root/autodl-tmp/ourDCGAN/SSIM/output_ssim_values.csv'  # 指定CSV文件路径
    # 写入CSV文件头部
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image SSIM Value'])

    for idx, img2 in enumerate(images2):
        max_ssim = 0.0
        for img1 in images1:
            # 添加window_size参数
            ssim_value = ssim_module(img1.unsqueeze(0), img2.unsqueeze(0), window_size=window_size).item()
            max_ssim = max(max_ssim, ssim_value)
        max_ssims.append(max_ssim)
        print(f"Maximum SSIM for image {idx + 1} from folder2: {max_ssim}")

        # 将max_ssim写入CSV文件
        with open(output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([max_ssim])

    avg_ssim = sum(max_ssims) / len(max_ssims)
    print(f"Average maximum SSIM between folders : {avg_ssim}")

    # 将平均值写入CSV文件
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Average SSIM Value'])
        writer.writerow([avg_ssim])

    return avg_ssim


# 使用示例
folder1_path = '/root/autodl-tmp/ourDCGAN/308_generate_images/clap' #生成图像路径
folder2_path = '/root/autodl-tmp/datasets/308/clap' #对应原始图像路径

average_ssim = calculate_max_ssim(folder2_path, folder1_path, target_size=(32, 32))