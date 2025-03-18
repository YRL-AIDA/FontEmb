import os.path
import random
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset


def label_to_vec(text):
    return torch.Tensor([float(text[0])])

def image_to_gray(image_left, image_right):
    grayscale_image_1 = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY) / 255.0
    grayscale_image_2 = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY) / 255.0
    return np.array([grayscale_image_1, grayscale_image_2]) # размерность канала

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
        label, i = self.__get_label_and_i_from_idx(idx)
        img_path = os.path.join(self.img_dir, label, f"image_{i}.png")
        image = Image.open(img_path)
        image_left = np.array(image.crop([0,0,120,60])) # левая картинка
        image_right = np.array(image.crop([120,0,240,60])) # правая картинка
        images = self.transform(image_left, image_right) if self.transform else np.array([image_left, image_right])
        if self.target_transform:
            label = self.target_transform(label)
        return torch.Tensor(images).unsqueeze(1) , label

    def __get_label_and_i_from_idx(self, idx):
        k = 0
        while (idx - self.counts[k]) >= 0:
            idx -= self.counts[k]
            k += 1
        return self.labels[k], idx




class SubCharCNNClassifier(nn.Module):
    def __init__(self):
        super(SubCharCNNClassifier, self).__init__()
        
        # сверточные слои
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # максимальный пулинг
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # полносвязный слой
        self.fc1 = nn.Linear(32 * 15 * 30, 128) 

        # функция активации
        self.relu = nn.ReLU()

    def forward(self, x):
        # применяем свертки и пулинг
        x = self.pool(self.relu(self.conv1(x)))  # (batch_size, 16, 30, 120)
        x = self.pool(self.relu(self.conv2(x)))  # (batch_size, 32, 15, 60)

        # делаем вектор одномерным для fc1
        x = x.view(x.size(0), -1)  # (batch_size, 32 * 15 * 60)

        # применяем полносвязный слой и relu
        x = self.relu(self.fc1(x))  # (batch_size, 128)

        return x


class ModelDiff(nn.Module):
    def __init__(self):
        super(ModelDiff, self).__init__()
        self.fc1 = nn.Linear(128 * 2, 128)  
        self.fc2 = nn.Linear(128, 1)  
        self.relu = nn.ReLU()

    def forward(self, emb_left, emb_right):
        x = torch.cat((emb_left, emb_right), dim=1)  # (batch_size, 256)
        x = self.relu(self.fc1(x))  # (batch_size, 128)
        x = self.fc2(x)  # (batch_size, 1)
        return x