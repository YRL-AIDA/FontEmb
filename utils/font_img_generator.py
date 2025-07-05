import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .text_generator import StringGenerator
from .based_generator import BasedGenerator

PATH_FONTS = os.path.join(os.path.dirname(__file__), '..', 'fonts')


class FontImgGenerator(BasedGenerator):
    def __init__(self, size_img=(40, 40), font_size=35):
        self.fonts = [os.path.join(PATH_FONTS, name) for name in os.listdir(PATH_FONTS)]

        self.image_size = size_img
        self.font_size = font_size
        self.intervals = [
            (-3, 3),  # отклонение по ширине
            (-3, 3)  # отклонение по высоте
        ]

    def draw_font(self, text, font_path, image_size, font_size):
        background = self.random_saturated_color(backcolor=True)
        image = Image.new('RGB', image_size, background)  # изображение с белым фоном
        draw = ImageDraw.Draw(image)

        font = ImageFont.truetype(font_path, font_size)

        position = self.random_position_with_constraints()

        draw.text(position, text, fill='black', font=font)

        # size = random.randint(10, 40)
        size = 40
        image = cv2.resize(np.array(image), (size, size), cv2.INTER_LANCZOS4)
        image = cv2.resize(image, (40, 40), cv2.INTER_LANCZOS4)

        return Image.fromarray(image.astype('uint8'), 'RGB')

    def generate_images(self, name_img, style=False, same_text=False):
        lang = random.choice(['rus', 'eng'])
        # одинаковый шрифт
        if style:
            font_path = random.choice(self.fonts)
            # font_name = os.path.basename(font_path).split('.')[0]
            images = []
            for i in range(2):
                text = StringGenerator.text_generator(lang)
                images.append(self.draw_font(text, font_path, self.image_size, self.font_size))
        # одинаковый текст
        elif same_text:
            text = StringGenerator.text_generator(lang)
            images = []
            for i in range(2):
                font_path = random.choice(self.fonts)
                # font_name = os.path.basename(font_path).split('.')[0]
                images.append(self.draw_font(text, font_path, self.image_size, self.font_size))
        # все разное
        else:
            images = []
            for i in range(2):
                font_path = random.choice(self.fonts)
                # font_name = os.path.basename(font_path).split('.')[0]
                text = StringGenerator.text_generator(lang)
                images.append(self.draw_font(text, font_path, self.image_size, self.font_size))
        final_image = Image.new('RGB', (images[0].width + images[1].width, images[1].height))
        final_image.paste(images[0], (0, 0))
        final_image.paste(images[1], (images[0].width, 0))
        final_image.save(name_img)
     