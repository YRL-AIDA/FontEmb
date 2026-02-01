import os.path
import random
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def label_to_vec(text):
    return torch.Tensor([float(text)])

IMAGE_LENGTH = 16000

class CharImageDataset(Dataset):
    def __init__(self, img_dir, target_transform=label_to_vec):
        self.img_dir = img_dir
        self.transform = T.Compose([
            T.Grayscale(),
            T.ToTensor()
        ])
        self.target_transform = target_transform
        self.image_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        if random.random() < 0.5:
            img_path = random.choice(self.image_paths)
            image = Image.open(img_path)

            crop1 = random.randint(0, IMAGE_LENGTH - 45)
            crop2 = random.randint(0, IMAGE_LENGTH - 45)

            image1 = image.crop((crop1, 0, crop1 + 40, 40))
            image2 = image.crop((crop2, 0, crop2 + 40, 40))
            label = '1'

        else:
            path1, path2 = random.sample(self.image_paths, 2)

            img1 = Image.open(path1)
            img2 = Image.open(path2)

            crop1 = random.randint(0, IMAGE_LENGTH - 45)
            crop2 = random.randint(0, IMAGE_LENGTH - 45)

            image1 = img1.crop((crop1, 0, crop1 + 40, 40))
            image2 = img2.crop((crop2, 0, crop2 + 40, 40))
            label = '0'

        image1 = self.transform(image1)
        image2 = self.transform(image2)
        if self.target_transform:
            label = self.target_transform(label)

        return (image1, image2), label



# class MultiDirCharImageDataset(CharImageDataset):
#     def __init(self, img_dirs, transform=image_to_gray, target_transform=label_to_vec):
#         self.img_dirs = img_dirs
#         self.labels = ['0', '1']
#         self.counts = [[len(os.listdir(os.path.join(img_dir, label))) for label in self.labels] for img_dir in
#                        self.img_dirs]
#         self.count = np.sum(self.counts)
#         self.transform = transform
#         self.target_transform = target_transform
#
#     def _get_image_path_idx(self, idx):
#         img_dir, label, i = self.__get_label_dir_and_i_from_idx(idx)
#         return os.path.join(img_dir, label, f"image_{i}.png"), label
#
#     def __get_label_dir_and_i_from_idx(self, idx):
#         # определить папку
#
#         count_in_dirs = [sum(cc) for cc in self.counts]
#
#         index_dir = 0
#         while (idx - count_in_dirs[index_dir]) >= 0:
#             idx -= count_in_dirs[index_dir]
#             index_dir += 1
#
#         # определить метки
#         k = 0
#         while (idx - self.counts[index_dir][k]) >= 0:
#             idx -= self.counts[index_dir][k]
#             k += 1
#
#         return self.img_dirs[index_dir], self.labels[k], idx