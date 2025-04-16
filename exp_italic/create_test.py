import sys
sys.path.append("..")
from create_dataset import create_dataset


if __name__ == '__main__':
    create_dataset("test", 1000)