import os.path
from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def label_to_vec(text):
    return torch.Tensor([float(text[0])])


def image_to_gray(image_left, image_right):
    grayscale_image_1 = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY) / 255.0
    grayscale_image_2 = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY) / 255.0
    return np.array([grayscale_image_1, grayscale_image_2])


class CharImageDataset(Dataset):
    def __init__(self, img_dir, transform=image_to_gray, target_transform=label_to_vec):
        
        self.img_dir = img_dir
        self.labels = ['0', '1']
        self.counts = [len(os.listdir(os.path.join(self.img_dir, label))) for label in self.labels]
        self.count = sum(self.counts)
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        img_path, label = self._get_image_path_idx(idx)
        image = Image.open(img_path)
        image_left = np.array(image.crop([0, 0, 40, 40])) # левая картинка
        image_right = np.array(image.crop([40, 0, 80, 40])) # правая картинка
        images = self.transform(image_left, image_right) if self.transform else np.array([image_left, image_right])
        if self.target_transform:
            label = self.target_transform(label)
        return torch.Tensor(images).unsqueeze(1), label

    def _get_image_path_idx(self, idx):
        label, i = self.__get_label_and_i_from_idx(idx)
        return os.path.join(self.img_dir, label, f"image_{i}.png"), label

    def __get_label_and_i_from_idx(self, idx):
        k = 0
        while (idx - self.counts[k]) >= 0:
            idx -= self.counts[k]
            k += 1
        return self.labels[k], idx
    
class MultiDirCharImageDataset(CharImageDataset):
    def __init__(self, img_dirs, transform=image_to_gray, target_transform=label_to_vec):
        self.img_dirs = img_dirs
        self.labels = ['0', '1']
        self.counts = [[len(os.listdir(os.path.join(img_dir, label))) for label in self.labels] for img_dir in self.img_dirs ]
        self.count = np.sum(self.counts)
        self.transform = transform
        self.target_transform = target_transform

    def _get_image_path_idx(self, idx):
        img_dir, label, i = self.__get_label_dir_and_i_from_idx(idx)
        return os.path.join(img_dir, label, f"image_{i}.png"), label

    def __get_label_dir_and_i_from_idx(self, idx):
        # определить папку
        
        count_in_dirs = [sum(cc) for cc in self.counts]

        index_dir = 0
        while (idx - count_in_dirs[index_dir]) >= 0:
            idx -= count_in_dirs[index_dir]
            index_dir += 1

        # определить метки
        k = 0
        while (idx - self.counts[index_dir][k]) >= 0:
            idx -= self.counts[index_dir][k]
            k += 1

        return self.img_dirs[index_dir], self.labels[k], idx
