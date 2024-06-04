import os
import cv2
import numpy as np
import random
from pathlib import Path


def add_gaussian_noise(image, mean=0, sigma=5):
    """在图像上添加高斯噪声"""
    gauss = np.random.normal(mean, sigma, image.shape).astype('uint8')
    noisy_image = cv2.add(image, gauss)
    return noisy_image


def add_salt_and_pepper_noise(image, salt_prob=0.1, pepper_prob=0.1):
    """在图像上添加椒盐噪声"""
    noisy_image = image.copy()
    total_pixels = image.shape[0] * image.shape[1]

    # 添加盐噪声
    num_salt = int(total_pixels * salt_prob)
    coords = [random.randint(0, i - 1) for i in image.shape]
    noisy_image[coords[0:num_salt]] = 255

    # 添加胡椒噪声
    num_pepper = int(total_pixels * pepper_prob)
    coords = [random.randint(0, i - 1) for i in image.shape]
    noisy_image[coords[0:num_pepper]] = 0

    return noisy_image


def process_dataset(input_folder, output_folder_gaussian, output_folder_salt_pepper):
    """处理数据集，生成带有高斯噪声和椒盐噪声的图像"""
    Path(output_folder_gaussian).mkdir(parents=True, exist_ok=True)
    Path(output_folder_salt_pepper).mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if image is not None:
            # 生成高斯噪声图像
            gaussian_noisy_image = add_gaussian_noise(image)
            gaussian_output_path = os.path.join(
                output_folder_gaussian, filename)
            cv2.imwrite(gaussian_output_path, gaussian_noisy_image)

            # 生成椒盐噪声图像
            salt_and_pepper_noisy_image = add_salt_and_pepper_noise(image)
            salt_pepper_output_path = os.path.join(
                output_folder_salt_pepper, filename)
            cv2.imwrite(salt_pepper_output_path, salt_and_pepper_noisy_image)

            print(f"Processed {filename}")


# 设置输入和输出文件夹路径
input_folder = "./test1"
output_folder_gaussian = "./tester_gaussian1/"
output_folder_salt_pepper = "./tester_salt_pepper1/"

# 处理数据集
process_dataset(input_folder, output_folder_gaussian,
                output_folder_salt_pepper)
