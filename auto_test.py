from create_train_dataset import create_dataset, FontImgGenerator, VariableFontImgGenerator
from train_model import (train,  
                         SubCharCNNClassifier, ModelDiff, ArticleCNNClassifier, CharImageDataset, 
                         MultiDirCharImageDataset)
from test_model import test_model

from torch import optim
from torch.nn import BCEWithLogitsLoss
import torch

device = torch.device('cuda:0' if torch.cuda.device_count() != 0 else 'cpu')
# Обучение
def train_model(params):
    optimizer = optim.Adam(list( params["model_cnn"].parameters()) + list(params["model_diff"].parameters()), lr=params["lr"])
    criterion = BCEWithLogitsLoss()
    train(optimizer, criterion, 
          params["dataset"], 
          params["model_cnn"],
          params["model_diff"],
          params["num_epochs"],
          params["batch_size"], 
          params["name"],
          device=device)


def tests(params, test_datasets):
    for name, test in test_datasets.items():
        res = test_model(params["model_cnn"], params["model_diff"], test, device=device)
        text = "="*40+f"""{params["name"]}
Test Dataset: {name}

Метрики:
Accuracy: {res["accuracy"]:.4f}
Precision: {res["precision"]:.4f}
Recall: {res["recall"]:.4f}
F1: {res["f1"]:.4f}
""" + "-"*40 + "\n"
        with open("result.txt", "a") as f:
            f.write(text + "\n") 


if __name__ == "__main__":
    # Создание наборов
    # COUNT_TRAIN_IMAGES = 100000
    # COUNT_TEST_IMAGES = 1000

    dataset_train_base = "train_dataset_base"
    dataset_train_vrbl = "train_dataset_vrbl"

    dataset_test_base = "test_dataset_base"
    dataset_test_vrbl = "test_dataset_vrbl"

    # create_dataset(dataset_train_base, FontImgGenerator(), COUNT_TRAIN_IMAGES)
    # create_dataset(dataset_train_vrbl, VariableFontImgGenerator(), COUNT_TRAIN_IMAGES)
    #
    # create_dataset(dataset_test_base, FontImgGenerator(), COUNT_TEST_IMAGES)
    # create_dataset(dataset_test_vrbl, VariableFontImgGenerator(), COUNT_TEST_IMAGES)

    # dataset_base = CharImageDataset(dataset_train_base)
    dataset_vrbl = CharImageDataset(dataset_train_vrbl)
    # dataset_mixed = MultiDirCharImageDataset([dataset_train_base, dataset_train_vrbl])

    test_base = CharImageDataset(dataset_test_base)
    test_vrbl = CharImageDataset(dataset_test_vrbl)
    test_mixed = MultiDirCharImageDataset([dataset_test_base, dataset_test_vrbl])


    # datasets = {"dataset_base": dataset_base, "dataset_vrbl": dataset_vrbl, "dataset_mixed": dataset_mixed}
    # models_class = {"sub_char_cnn": SubCharCNNClassifier, "article_cnn": ArticleCNNClassifier}

    datasets = {"dataset_vrbl": dataset_vrbl}
    models_class = {"sub_char_cnn": SubCharCNNClassifier}

    exps = [
    {
        "name": "exp" + " " + name_ds + " " + name_m,
        "model_cnn": model_cnn(),
        "model_diff": ModelDiff(),
        "dataset": dataset,
        "num_epochs": 30,
        "batch_size": 256,
        "lr": 0.0005,
    }  for name_ds, dataset in datasets.items() for name_m, model_cnn in models_class.items() ] 
    
    for params in exps:
        train_model(params)
        tests(params, {"test_base": test_base, "test_vrbl": test_vrbl, "test_mixed": test_mixed})