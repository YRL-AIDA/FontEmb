from utils import FontImgGenerator  # Шрифты без свойств
from utils import VariableFontImgGenerator
import os

if __name__ == '__main__':
    COUNT_IMAGES = 1000
    COUNT_IMAGES_0 = COUNT_IMAGES // 2
    COUNT_IMAGES_1 = COUNT_IMAGES // 2

    font_generator = FontImgGenerator()
    font_generator1 = VariableFontImgGenerator()
    os.mkdir('test2')
    path_0 = os.path.join('test2', '0')
    path_1 = os.path.join('test2', '1')
    os.mkdir(path_0)
    os.mkdir(path_1)

    for i in range(COUNT_IMAGES_0 // 2):
        if i < COUNT_IMAGES_0 // 4:
            font_generator.generate_images(os.path.join(path_0, f'image_normal_{i}.png'))
            font_generator1.generate_images(os.path.join(path_0, f'image_variable_{i}.png'))
        else:
            font_generator.generate_images(os.path.join(path_0,  f'image_normal_{i}.png'), same_text=True)
            font_generator1.generate_images(os.path.join(path_0, f'image_variable_{i}.png'), same_text=True)
    for i in range(COUNT_IMAGES_1 // 2):
        font_generator.generate_images(os.path.join(path_1,  f'image_normal_{i}.png'), style=True)
        font_generator1.generate_images(os.path.join(path_1, f'image_variable_{i}.png'), style=True)