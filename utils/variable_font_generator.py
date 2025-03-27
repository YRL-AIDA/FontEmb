import os
import random
from PIL import Image, ImageDraw, ImageFont
from text_generator import StringGenerator

PATH_FONTS = os.path.join(os.path.dirname(__file__), '..', 'variable_fonts')


class VariableFontImgGenerator:
    def __init__(self, size_img=(60, 60), font_size=40):
        self.fonts = [os.path.join(PATH_FONTS, name) for name in os.listdir(PATH_FONTS)]
        self.image_size = size_img
        self.font_size = font_size
        self.intervals = [
            (-10, 10),  # отклонение по ширине
            (-20, 10)  # отклонение по высоте
        ]

        self.weights = [400, 800]  # жирность
        self.backgrounds = ['yellow', 'green', 'red', 'blue', None]  # фон
        self.strike = [True, False]  # зачеркивание
        self.underline = [True, False]  # подчеркивание

    def random_position_with_constraints(self):
        x_interval, y_interval = self.intervals
        x = random.randint(x_interval[0], x_interval[1])
        y = random.randint(y_interval[0], y_interval[1])
        return (x, y)

    def draw_variable_font(self, text, font_path, image_size, font_size, weight, background, strike, underline):
        bg_color = background if background else 'white'
        image = Image.new('RGB', image_size, bg_color)
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

        return image

    def generate_random_style(self):
        # случайная комбинация стилей
        font_path = random.choice(self.fonts)
        weight = random.choice(self.weights)
        background = random.choice(self.backgrounds)
        strike = random.choice(self.strike)
        underline = random.choice(self.underline)
        return font_path, weight, background, strike, underline

    def generate_images(self, name_img, style=False, same_text=False):
        lang = random.choice(['rus', 'eng'])

        # одинаковый шрифт
        if style:
            font_path, weight, _, _, _ = self.generate_random_style()
            images = []
            for i in range(2):
                text = StringGenerator.text_generator(lang)

                # не берем в расчет фон, зачеркивание и подчеркивание
                background = random.choice(self.backgrounds)
                strike = random.choice(self.strike)
                underline = random.choice(self.underline)

                image = self.draw_variable_font(text, font_path, self.image_size, self.font_size, weight, background,
                                                strike, underline)
                images.append(image)

        # одинаковый текст
        elif same_text:
            text = StringGenerator.text_generator(lang)
            images = []
            for i in range(2):
                font_path, weight, background, strike, underline = self.generate_random_style()
                image = self.draw_variable_font(text, font_path, self.image_size, self.font_size, weight, background,
                                                strike, underline)
                images.append(image)

        # все разное
        else:
            images = []
            for i in range(2):
                font_path, weight, background, strike, underline = self.generate_random_style()
                text = StringGenerator.text_generator(lang)
                image = self.draw_variable_font(text, font_path, self.image_size, self.font_size, weight, background,
                                                strike, underline)
                images.append(image)

        final_image = Image.new('RGB', (images[0].width + images[1].width, images[1].height))
        final_image.paste(images[0], (0, 0))
        final_image.paste(images[1], (images[0].width, 0))
        final_image.save(name_img)

# Проверка
generator = VariableFontImgGenerator()
generator.generate_images('output.png', style=True)  # одинаковый шрифт
generator.generate_images('output.png', same_text=True)  # одинаковый текст
generator.generate_images('output.png')  # все разное