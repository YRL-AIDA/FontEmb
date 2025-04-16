import sys, os
sys.path.append("..")
from utils import ItalicImgGenerator
import os

def create_dataset(name_dataset,count_image):
    COUNT_IMAGES_0 = count_image//2
    COUNT_IMAGES_1 = count_image//2

    font_generator = ItalicImgGenerator()
    os.mkdir(name_dataset)
    path_0 = os.path.join(name_dataset, '0')
    path_1 = os.path.join(name_dataset, '1')
    os.mkdir(path_0)
    os.mkdir(path_1)
    
    for i in range(COUNT_IMAGES_0):
        font_generator.generate_images(os.path.join(path_0, f'image_{i}.png'), italic=False)
    for i in range(COUNT_IMAGES_1):
        font_generator.generate_images(os.path.join(path_1, f'image_{i}.png'), italic=True)

if __name__ == '__main__':
    create_dataset(name_dataset='dataset',count_image=10000)