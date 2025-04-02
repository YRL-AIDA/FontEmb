import os
import random
from PIL import Image, ImageDraw, ImageFont
from text_generator import StringGenerator

PATH_FONTS = os.path.join(os.path.dirname(__file__), '..', 'variable_fonts')


class BoldImgGenerator:
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

    def random_saturated_color(self, backcolor=False):
        # Генерация случайных значений для R, G, B
        r = random.randint(0, 127)
        g = random.randint(0, 127)
        b = random.randint(0, 127)

        if max(r, g, b) - min(r, g, b) < 50:
            if random.choice([True, False]):
                r = random.choice([0, 127])
            else:
                g = random.choice([0, 127])

        return (r, g, b) if not backcolor else (255 - r // 4, 255 - g // 4, 255 - b // 4)

    def random_position_with_constraints(self):
        x_interval, y_interval = self.intervals
        x = random.randint(x_interval[0], x_interval[1])
        y = random.randint(y_interval[0], y_interval[1])
        return (x, y)

    def draw_variable_font(self, text, font_path, image_size, font_size, weight, strike, underline):
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

        return image

    def generate_random_style(self):
        # случайная комбинация стилей
        font_path = random.choice(self.fonts)
        strike = random.choice(self.strike)
        underline = random.choice(self.underline)
        return font_path, strike, underline

    def generate_images(self, name_img, bold=False):
        lang = random.choice(['rus', 'eng'])

        # жирные
        if bold:
            weight = 800
            text = StringGenerator.text_generator(lang)
            font_path, strike, underline = self.generate_random_style()
            image = self.draw_variable_font(text, font_path, self.image_size, self.font_size, weight,
                                            strike, underline)
            print(font_path)

        # нежирные
        else:
            weight = 400
            font_path, strike, underline = self.generate_random_style()
            text = StringGenerator.text_generator(lang)
            image = self.draw_variable_font(text, font_path, self.image_size, self.font_size, weight,
                                            strike, underline)

        image.save(name_img)


# Проверка
generator = BoldImgGenerator()
generator.generate_images('output.png', bold=True)  # жирные
# generator.generate_images('output.png', bold=False)  # нежирные
# generator.generate_images('output.png')  # все разное