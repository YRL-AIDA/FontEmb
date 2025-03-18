import random

class StringGenerator:
    @staticmethod
    def text_generator(lang):
        string = ''
        alphabet_rus_small = 'абвгдеёжзийклмопрстфхцшщьыъэюя'
        alphabet_rus_big = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
        alphabet_eng_small = 'abcdefghijklmnopqrstuvwxyz'
        alphabet_eng_big = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
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

def text_generator(lang):
    return StringGenerator().text_generator(lang)