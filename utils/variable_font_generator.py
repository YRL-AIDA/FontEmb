import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .text_generator import StringGenerator
from .based_generator import BasedGenerator

PATH_FONTS = os.path.join(os.path.dirname(__file__), '..', 'variable_fonts')


class VariableFontImgGenerator(BasedGenerator):
    def __init__(self, size_img=(40, 40), font_size=35):
        self.fonts = [os.path.join(PATH_FONTS, name) for name in os.listdir(PATH_FONTS)]
        self.image_size = size_img
        self.font_size = font_size
        self.intervals = [
            (-3, 3),  # отклонение по ширине
            (-3, 3)  # отклонение по высоте
        ]

        self.weights = [400, 800]  # жирность
        self.strike = [True, False]  # зачеркивание
        self.underline = [True, False]  # подчеркивание

    def draw_font(self, text, font_path, image_size, font_size, weight, strike, underline):
        background = self.random_saturated_color(backcolor=True)
        image = Image.new('RGB', image_size, background)
        draw = ImageDraw.Draw(image)

        font = ImageFont.truetype(font_path, font_size)

        # жирность
        if hasattr(font, 'set_variation_by_axes'):
            font.set_variation_by_axes([weight])

        position = self.random_position_with_constraints()

        bbox = draw.textbbox(position, text, font=font)

        draw.text(position, text, fill='black', font=font)

        # зачеркивание текста
        if strike:
            strike_position = (bbox[0], (bbox[1] + bbox[3]) // 2)
            draw.line([bbox[0], strike_position[1], bbox[2], strike_position[1]], fill='black', width=2)

        # подчеркивание текста
        if underline:
            underline_position = (bbox[0], bbox[3] + 5)  # Положение линии под текстом (5 пикселей ниже)
            draw.line([bbox[0], underline_position[1], bbox[2], underline_position[1]], fill='black', width=2)

        size = random.randint(10, 40)
        image = cv2.resize(np.array(image), (size, size), cv2.INTER_LANCZOS4)
        image = cv2.resize(image, (40, 40), cv2.INTER_LANCZOS4)

        return Image.fromarray(image.astype('uint8'), 'RGB')

    def generate_random_style(self):
        # случайная комбинация стилей
        font_path = random.choice(self.fonts)
        weight = random.choice(self.weights)
        # background = random.choice(self.backgrounds)
        strike = random.choice(self.strike)
        underline = random.choice(self.underline)
        return font_path, weight, strike, underline

    def generate_images(self, name_img, style=False, same_text=False):
        lang = random.choice(['rus', 'eng'])

        # одинаковый шрифт (не учитываем фон)
        if style:
            font_path, weight, strike, underline = self.generate_random_style()
            images = []
            for i in range(2):
                text = StringGenerator.text_generator(lang)

                image = self.draw_font(text, font_path, self.image_size, self.font_size, weight,
                                                strike, underline)
                images.append(image)

        # одинаковый текст
        elif same_text:
            text = StringGenerator.text_generator(lang)
            images = []
            for i in range(2):
                font_path, weight, strike, underline = self.generate_random_style()
                image = self.draw_font(text, font_path, self.image_size, self.font_size, weight,
                                                strike, underline)
                images.append(image)

        # все разное
        else:
            images = []
            for i in range(2):
                font_path, weight, strike, underline = self.generate_random_style()
                text = StringGenerator.text_generator(lang)
                image = self.draw_font(text, font_path, self.image_size, self.font_size, weight,
                                                strike, underline)
                images.append(image)

        final_image = Image.new('RGB', (images[0].width + images[1].width, images[1].height))
        final_image.paste(images[0], (0, 0))
        final_image.paste(images[1], (images[0].width, 0))
        final_image.save(name_img)

# Проверка
# generator = VariableFontImgGenerator()
# generator.generate_images('output.png', style=True)  # одинаковый шрифт
# generator.generate_images('output.png', same_text=True)  # одинаковый текст
# generator.generate_images('output.png')  # все разное