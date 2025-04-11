from utils import FontImgGenerator # Шрифты без свойств
from utils import VariableFontImgGenerator
import os

def create_dataset(name_dataset, generator, count):
    COUNT_IMAGES_0 = count//2
    COUNT_IMAGES_1 = count//2
    os.mkdir(name_dataset)
    path_0 = os.path.join(name_dataset, '0')
    path_1 = os.path.join(name_dataset, '1')
    os.mkdir(path_0)
    os.mkdir(path_1)
    
    for i in range(COUNT_IMAGES_0):
        if i < COUNT_IMAGES_0//2:
            generator.generate_images(os.path.join(path_0, f'image_{i}.png'))
        else:
            generator.generate_images(os.path.join(path_0, f'image_{i}.png'), same_text=True)
    for i in range(COUNT_IMAGES_1):
        generator.generate_images(os.path.join(path_1, f'image_{i}.png'), style=True)

if __name__ == '__main__':
    COUNT_IMAGES = 100
    font_generator = FontImgGenerator()
    create_dataset('train_dataset_', font_generator, COUNT_IMAGES)

