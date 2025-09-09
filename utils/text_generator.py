import random

class StringGenerator:

    alphabet_rus_small = 'абвгдеёжзийклмопрстфхцшщьыъэюя'
    alphabet_rus_big = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
    alphabet_eng_small = 'abcdefghijklmnopqrstuvwxyz'
    alphabet_eng_big = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    @staticmethod
    def text_generator(lang):
        alphabet_rus_small = StringGenerator.alphabet_rus_small
        alphabet_rus_big = StringGenerator.alphabet_rus_big
        alphabet_eng_small = StringGenerator.alphabet_eng_small
        alphabet_eng_big = StringGenerator.alphabet_eng_big
        string = ''
        choice = random.randint(1, 3)
        if lang == 'rus':
            if choice == 1:
                for i in range(3):
                    string += random.choice(alphabet_rus_small)
            elif choice == 2:
                for i in range(3):
                    string += random.choice(alphabet_rus_big)
            elif choice == 3:
                string = random.choice(alphabet_rus_big) + random.choice(alphabet_rus_small) + random.choice(alphabet_rus_small)
        elif lang == 'eng':
            if choice == 1:
                for i in range(3):
                    string += random.choice(alphabet_eng_small)
            elif choice == 2:
                for i in range(3):
                    string += random.choice(alphabet_eng_big)
            elif choice == 3:
                string = random.choice(alphabet_eng_big) + random.choice(alphabet_eng_small) + random.choice(alphabet_eng_small)
        return string

    @staticmethod
    def contrastive_text_generator(lang):
        string = ''
        alphabet_rus_small = StringGenerator.alphabet_rus_small
        alphabet_rus_big = StringGenerator.alphabet_rus_big
        alphabet_eng_small = StringGenerator.alphabet_eng_small
        alphabet_eng_big = StringGenerator.alphabet_eng_big
        if lang == 'rus':
            for i in range(1000):
                string += random.choice([random.choice(alphabet_rus_small), random.choice(alphabet_rus_small), random.choice(alphabet_rus_big)])
        elif lang == 'eng':
            for i in range(1000):
                string += random.choice([random.choice(alphabet_eng_small), random.choice(alphabet_eng_small), random.choice(alphabet_eng_big)])
        return string


def text_generator(lang):
    return StringGenerator().contrastive_text_generator(lang)