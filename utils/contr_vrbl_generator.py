import os
import random
import cv2
import numpy as np
import itertools
from PIL import Image, ImageDraw, ImageFont
from .text_generator import StringGenerator
from .based_generator import BasedGenerator

PATH_FONTS = os.path.join(os.path.dirname(__file__), '..', 'variable_fonts')


class ContrastiveVariableFontGenerator(BasedGenerator):
    def __init__(self, size_img=(16000, 40), font_size=35):
        self.fonts = [os.path.join(PATH_FONTS, name) for name in os.listdir(PATH_FONTS)]
        self.image_size = size_img
        self.font_size = font_size
        # self.intervals = [
        #     (-3, 3),  # отклонение по ширине
        #     (-3, 3)  # отклонение по высоте
        # ]

        self.weights = [400, 800]  # жирность
        self.strike = [True, False]  # зачеркивание
        self.underline = [True, False]  # подчеркивание

    def draw_font(self, text, font_path, image_size, font_size, weight, strike, underline):
        background = self.random_saturated_color(backcolor=True)
        image = Image.new('RGB', image_size, background)
        draw = ImageDraw.Draw(image)

        font = ImageFont.truetype(font_path, font_size)

        if hasattr(font, 'set_variation_by_axes'):
            font.set_variation_by_axes([weight])

        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_width = right - left
        text_height = bottom - top

        x = (image_size[0] - text_width) // 2 - left
        y = (image_size[1] - text_height) // 2 - top

        draw.text((x, y), text, fill='black', font=font)

        bbox = draw.textbbox((x, y), text, font=font)

        # зачеркивание текста
        if strike:
            mid_y = (bbox[1] + bbox[3]) // 2
            draw.line([bbox[0], mid_y, bbox[2], mid_y], fill='black', width=1)

        # подчеркивание текста
        if underline:
            underline_y = bbox[3] + 1
            draw.line([bbox[0], underline_y, bbox[2], underline_y], fill='black', width=1)

        # size = random.randint(10, 40)
        # size = 40
        # image = cv2.resize(np.array(image), (size, size), cv2.INTER_LANCZOS4)
        # image = cv2.resize(image, (40, 40), cv2.INTER_LANCZOS4)

        return image

    def generate_random_style(self):
        # случайная комбинация стилей
        font_path = random.choice(self.fonts)
        weight = random.choice(self.weights)
        # background = random.choice(self.backgrounds)
        strike = random.choice(self.strike)
        underline = random.choice(self.underline)
        return font_path, weight, strike, underline

    def generate_images(self, name_img):
        lang = random.choice(['rus', 'eng'])
        a = {
            True: '1',
            False: '0',
            400: '0',
            800: '1',

        }
        font_path, weight, strike, underline = self.generate_random_style()
        text = StringGenerator.contrastive_text_generator(lang)

        image = self.draw_font(text, font_path, self.image_size, self.font_size, weight,
                                   strike, underline)
        image.save(f'{name_img}_{"1" if "italic" in font_path else "0"}_{a[weight]}_{a[strike]}_{a[underline]}.png')
        
    def generate_contrastive_styles(self):
        font_styles = list(itertools.product(self.fonts, self.weights, self.strike, self.underline))
        langs = ['rus', 'eng']
        for lang in langs:
            for i, style in enumerate(font_styles):
                text = StringGenerator.contrastive_text_generator(lang)
                image = self.draw_font(text, style[0], self.image_size, self.font_size, style[1],style[2], style[3])
                image.save(f'contr_dataset/image_{i}_{lang}.png')



# self.fonts = [os.path.join(PATH_FONTS, name) for name in os.listdir(PATH_FONTS)]
#         self.image_size = size_img
#         self.font_size = font_size
#         # self.intervals = [
#         #     (-3, 3),  # отклонение по ширине
#         #     (-3, 3)  # отклонение по высоте
#         # ]

#         self.weights = [400, 800]  # жирность
#         self.strike = [True, False]  # зачеркивание
#         self.underline = [True, False]  # подчеркивание

# Проверка
# generator = ContrastiveVariableFontGenerator()
# generator.generate_images('output.png')
# generator.generate_images('output.png', style=True)  # одинаковый шрифт
# generator.generate_images('output.png', same_text=True)  # одинаковый текст
# generator.generate_images('output.png')  # все разное