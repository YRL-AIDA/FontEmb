import numpy as np
from PIL import Image
from model_architecture.model import ModelDiff, SubCharCNNClassifier
from model_architecture.article_model import ArticleCNNClassifier
from dataset_class import CharImageDataset 
from dataset_class import image_to_gray
import torch
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device('cuda:0' if torch.cuda.device_count() != 0 else 'cpu')

def test_model(model_cnn, model_diff, dataset, device, batch_size=10):
    true_labels = []
    pred_labels = []
    len_dataset = len(dataset)
    batchs = [[j for j in range(i, i+batch_size)] for i in range(0, len_dataset, batch_size)]
    for index in batchs:
        batch = [dataset[i] for i in index]
        left_img = torch.cat([b[0][0].unsqueeze(0) for b in batch], dim=0).to(device)
        right_img = torch.cat([b[0][1].unsqueeze(0) for b in batch], dim=0).to(device)
        targets = torch.cat([b[1].unsqueeze(0) for b in batch], dim=0).to(device)
        
        font_emb_left = model_cnn(left_img).to(device)  # (batch_size, 128)
        font_emb_right = model_cnn(right_img).to(device)  # (batch_size, 128)

        # выход 
        sameness = model_diff(font_emb_left, font_emb_right).to(device)  # (batch_size, 1)
        # Приводим метки к нужной форме
        targets = targets.view(-1, 1).to(device)
        pred = torch.sigmoid(sameness)
        true_labels += [t[0] for t in targets.tolist()]
        pred_labels += [1 if p[0] > 0.5 else 0 for p in pred.tolist() ]

    # print()
    # print(true_labels)
    # print(pred_labels)
    
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    return {"f1": f1, "accuracy": accuracy, "precision": precision, "recall": recall}


def classifier(model_cnn, model_diff, char_left, char_right):
    gray_image = image_to_gray(char_left, char_right)
    data_left = torch.Tensor(gray_image[0]).unsqueeze(0).unsqueeze(0)
    data_right = torch.Tensor(gray_image[1]).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        emb_left = model_cnn(data_left)
        emb_right = model_cnn(data_right)
        rez = model_diff(emb_left, emb_right)
        probability = torch.sigmoid(rez)
    return probability.item()


if __name__ == "__main__":
    name_model1 = "test_exp_cnn.pt"
    name_diff_model = "test_exp_diff.pt"

    model1 = SubCharCNNClassifier()
    # model1 = ArticleCNNClassifier()
    model1.load_state_dict(torch.load(name_model1, map_location=device))
    model1.eval()

    diff_model = ModelDiff()
    diff_model.load_state_dict(torch.load(name_diff_model, map_location=device))
    diff_model.eval()
    test_dirs = ['test_dataset_base']
    results_file = "test_results.txt"

    with open(results_file, 'a', encoding='utf-8') as f:
        for test_dir in test_dirs:
            test_base = CharImageDataset(test_dir)
            res = test_model(model1, diff_model, test_base, device)
            text = "="*40+f"""
Датасет: {test_dir}
Модель 1: {name_model1}
Модель Diff: {name_diff_model}

Метрики:
Accuracy: {res["accuracy"]}
Precision: {res["precision"]}
Recall: {res["recall"]}
F1: {res["f1"]}
""" + "-"*40 + "\n"
            print(text)
            f.write(text)