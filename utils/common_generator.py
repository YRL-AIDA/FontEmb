import os
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTCollection

from .text_generator import StringGenerator
from .based_generator import BasedGenerator

PATH_FONTS = os.path.join(os.path.dirname(__file__), '..', 'fonts')


class CommonGenerator(BasedGenerator):
    def __init__(self, size_img=(16000, 40), font_size=35):
        self.fonts = [os.path.join(PATH_FONTS, name) for name in os.listdir(PATH_FONTS)]
        self.image_size = size_img
        self.font_size = font_size

    def draw_font(self, text, font_path, image_size, font_size, index=0):
        background = self.random_saturated_color(backcolor=True)
        image = Image.new('RGB', image_size, background)
        draw = ImageDraw.Draw(image)

        font = ImageFont.truetype(font_path, font_size, index=index)

        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_width = right - left
        text_height = bottom - top

        x = (image_size[0] - text_width) // 2 - left
        y = (image_size[1] - text_height) // 2 - top

        draw.text((x, y), text, fill='black', font=font)

        return image

    def generate_images(self):
        lang = 'all'

        for font_path in self.fonts:
            if font_path.endswith('.ttc'):
                ttc = TTCollection(font_path)

                for i in range(len(ttc)):

                    font_name = f"{os.path.splitext(os.path.basename(font_path))[0]}_{i}"
                    text = StringGenerator.contrastive_text_generator(lang)
                    image = self.draw_font(text, font_path, self.image_size, self.font_size, index=i)

                    filename = f"train/{font_name}.png"
                    image.save(filename)

            else:
                font_name = os.path.splitext(os.path.basename(font_path))[0]
                text = StringGenerator.contrastive_text_generator(lang)
                image = self.draw_font(text, font_path, self.image_size, self.font_size)

                filename = f"train/{font_name}.png"
                image.save(filename)

