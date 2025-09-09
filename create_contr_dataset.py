
from utils import ContrastiveVariableFontGenerator
import os


def create_dataset(name_dataset, generator):
    os.makedirs(name_dataset, exist_ok=True)
        
    generator.generate_contrastive_styles()

if __name__ == '__main__':
    font_generator = ContrastiveVariableFontGenerator()
    create_dataset('contr_dataset', font_generator)

