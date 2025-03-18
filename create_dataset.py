from utils import FontImgGenerator
import os

if __name__ == '__main__':
    COUNT_IMAGES = 10000
    COUNT_IMAGES_0 = COUNT_IMAGES//2
    COUNT_IMAGES_1 = COUNT_IMAGES//2

    font_generator = FontImgGenerator()
    os.mkdir('dataset')
    path_0 = os.path.join('dataset', '0')
    path_1 = os.path.join('dataset', '1')
    os.mkdir(path_0)
    os.mkdir(path_1)
    
    for i in range(COUNT_IMAGES_0):
        if i < COUNT_IMAGES_0//2:
            font_generator.generate_images(os.path.join(path_0, f'image_{i}.png'))
        else:
            font_generator.generate_images(os.path.join(path_0, f'image_{i}.png'), same_text=True)
    for i in range(COUNT_IMAGES_1):
        font_generator.generate_images(os.path.join(path_1, f'image_{i}.png'), style=True)