import os
import random
from PIL import Image, ImageDraw, ImageFont
from .text_generator import StringGenerator

PATH_FONTS = os.path.join(os.path.dirname(__file__), '..', 'fonts')
class FontImgGenerator:
    def __init__(self, size_img=(120, 60), font_size=40):
        self.fonts = [os.path.join(PATH_FONTS, name) for name in os.listdir(PATH_FONTS)]

        self.image_size = size_img
        self.font_size = font_size
        self.intervals = [
            (-10, 10),  # отклонение по ширине
            (-20, 10)  # отклонение по высоте
        ]

    def random_position_with_constraints(self):
        # разделяем интервалы для ширины (x) и высоты (y)
        x_interval, y_interval = self.intervals

        # генерация случайной позиции по ширине
        x = random.randint(x_interval[0], x_interval[1])

        # генерация случайной позиции по высоте
        y = random.randint(y_interval[0], y_interval[1])

        return (x, y)

    def draw_font(self, text, font_path, image_size, font_size):
        image = Image.new('RGB', image_size, 'white')  # изображение с белым фоном
        draw = ImageDraw.Draw(image)

        font = ImageFont.truetype(font_path, font_size)

        position = self.random_position_with_constraints()

        draw.text(position, text, fill='black', font=font)

        return image

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
     