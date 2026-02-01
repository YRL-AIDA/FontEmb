
from utils import ContrastiveVariableFontGenerator
from utils import CommonGenerator
import os


def create_dataset(name_dataset, generator):
    os.makedirs(name_dataset, exist_ok=True)
        
    # generator.generate_contrastive_styles()
    generator.generate_images()


if __name__ == '__main__':
    # font_generator = ContrastiveVariableFontGenerator()
    # create_dataset('contr_dataset', font_generator)

    font_generator = CommonGenerator()
    create_dataset('train', font_generator)

