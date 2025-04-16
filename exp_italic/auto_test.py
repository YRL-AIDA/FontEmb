import os
from create_dataset import create_dataset
from fine_tuning import train, SubCharCNNClassifier, ArticleCNNClassifier, ItalicTask, CharImageDataset
from test_model import test_model
import torch
from torch import optim
from torch.nn import BCEWithLogitsLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    NAME_TRAIN_DATASET = "dataset"
    NAME_TEST_DATASET = "test"
    create_dataset(NAME_TRAIN_DATASET,10000)
    create_dataset(NAME_TEST_DATASET, 1000)

    train_dataset = CharImageDataset("dataset/")
    test_dataset = CharImageDataset("test/")

    LOG_FILE = "log.train.txt"

    MODEL_EMB_FILE_SUBCHAR = os.path.join("..", "exp dataset_mixed sub_char_cnn_cnn.pt")
    MODEL_EMB_FILE_ARTICLE = os.path.join("..", "exp dataset_mixed article_cnn_cnn.pt")
    num_epochs = 100
    

    model_sub_char_cnn_emb = SubCharCNNClassifier().to(device)
    model_sub_char_cnn_emb.load_state_dict(torch.load(MODEL_EMB_FILE_SUBCHAR, map_location=device))
    model_sub_char_cnn_emb.eval()

    model_article_cnn_emb = ArticleCNNClassifier().to(device)
    model_article_cnn_emb.load_state_dict(torch.load(MODEL_EMB_FILE_ARTICLE, map_location=device))
    model_article_cnn_emb.eval()

    models_emb = [{
                    "model_name": "sub_char",
                    "model_emb": model_sub_char_cnn_emb,
                    "model_italic": ItalicTask().to(device),
                    "path_model": MODEL_EMB_FILE_SUBCHAR
                  }, 
                  {
                    "model_name": "article",
                    "model_italic": ItalicTask().to(device),
                    "model_emb": model_article_cnn_emb,
                    "path_model": MODEL_EMB_FILE_ARTICLE
                  }
    ]
    
    for param in models_emb:
        criterion = BCEWithLogitsLoss().to(device) 
        optimizer = optim.Adam(list(param["model_italic"].parameters()), lr=0.0025)
        


        train(param["model_emb"], param["model_italic"], train_dataset, optimizer, criterion, 
            num_epochs=100,  log_file=LOG_FILE, name_model=param["model_name"]+"_italic.pt",)


        res = test_model(param["model_emb"], param["model_italic"], test_dataset, device)
        text = "="*40+f"""
Модель: {param['model_name']}

Метрики:
Accuracy: {res["accuracy"]}
Precision: {res["precision"]}
Recall: {res["recall"]}
F1: {res["f1"]}
""" + "-"*40 + "\n"
        print(text)
