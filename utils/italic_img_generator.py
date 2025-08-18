import os
import random
from PIL import Image, ImageDraw, ImageFont
from .text_generator import StringGenerator
import cv2
import numpy as np
from .based_generator import BasedGenerator

PATH_FONTS = os.path.join(os.path.dirname(__file__), '..', 'variable_fonts')


class ItalicImgGenerator(BasedGenerator):
    def __init__(self, size_img=(40, 40), font_size=35):
        self.fonts_italic = [os.path.join(PATH_FONTS, name) for name in os.listdir(PATH_FONTS) if "italic" in name ]
        self.fonts_noitalic = [os.path.join(PATH_FONTS, name) for name in os.listdir(PATH_FONTS) if not "italic" in name]
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
        fs = font_size + random.randint(-10, 10)  # Размер шрифта с возможным отклонением
        font = ImageFont.truetype(font_path, fs)

        # жирность
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
        size = 40
        image = cv2.resize(np.array(image), (size, size), cv2.INTER_LANCZOS4)
        image = cv2.resize(image, (40, 40), cv2.INTER_LANCZOS4)

        return  Image.fromarray(image.astype('uint8'), 'RGB')

    def generate_random_style(self, italic):
        # случайная комбинация стилей
        font_path = random.choice(self.fonts_italic) if italic else random.choice(self.fonts_noitalic)
        strike = random.choice(self.strike)
        weight = random.choice(self.weights)
        underline = random.choice(self.underline)
        return font_path, strike, weight, underline

    def generate_images(self, name_img, italic=False):
        lang = random.choice(['rus', 'eng'])

        # курсив
        if italic:
            text = StringGenerator.text_generator(lang)
            font_path, strike, weight, underline = self.generate_random_style(italic)
            image = self.draw_font(text, font_path, self.image_size, self.font_size, weight,
                                            strike, underline)
            # print(font_path)

        # не курсив
        else:
            font_path, strike, weight, underline = self.generate_random_style(italic)
            text = StringGenerator.text_generator(lang)
            image = self.draw_font(text, font_path, self.image_size, self.font_size, weight,
                                            strike, underline)

        image.save(name_img)


# Проверка
#generator = BoldImgGenerator()
#generator.generate_images('output.png', bold=True)  # жирные
# generator.generate_images('output.png', bold=False)  # нежирные
# generator.generate_images('output.png')  # все разное