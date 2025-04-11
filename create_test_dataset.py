from create_train_dataset import create_dataset, FontImgGenerator, VariableFontImgGenerator

if __name__ == '__main__':
    COUNT_IMAGES = 1000
    font_generator = FontImgGenerator()
    create_dataset("test", font_generator, COUNT_IMAGES)