import sys, os
sys.path.append("..")
import numpy as np
from PIL import Image
from model_architecture.model import  SubCharCNNClassifier
import torch
from fine_tuning import ItalicTask, image_to_gray, CharImageDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device('cuda:0' if torch.cuda.device_count() != 0 else 'cpu')
# device = torch.device('cpu')

def test_model(model_emb, model_italic, dataset, device, batch_size=10):
    true_labels = []
    pred_labels = []
    len_dataset = len(dataset)
    batchs = [[j for j in range(i, i+batch_size)] for i in range(0, len_dataset, batch_size)]
    for index in batchs:
        batch = [dataset[i] for i in index]
        imgs = torch.cat([b[0].unsqueeze(0) for b in batch], dim=0).to(device)
        targets = torch.cat([b[1].unsqueeze(0) for b in batch], dim=0).to(device)
        
        font_emb_img = model_emb(imgs).to(device)  # (batch_size, 128)

        # выход 
        italic = model_italic(font_emb_img).to(device)  # (batch_size, 1)

        # Приводим метки к нужной форме
        targets = targets.view(-1, 1).to(device)  
        pred = torch.sigmoid(italic)
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


if __name__ == "__main__":
    name_model_emb = os.path.join("..", "model.pt")
    name_italic_model = "model_italic.pt"

    model_emb= SubCharCNNClassifier().to(device)
    model_emb.load_state_dict(torch.load(name_model_emb, map_location=device))
    model_emb.eval()

    italic_model = ItalicTask().to(device)
    italic_model.load_state_dict(torch.load(name_italic_model, map_location=device))
    italic_model.eval()

    test_dirs = ['test']
    results_file = "test_results.txt"


    with open(results_file, 'a', encoding='utf-8') as f:
        for test_dir in test_dirs:
            test_base = CharImageDataset(test_dir)
            res = test_model(model_emb, italic_model, test_base, device)
            text = "="*40+f"""
Датасет: {test_dir}
Модель Emb: {name_model_emb}
Модель Italic: {name_italic_model}

Метрики:
Accuracy: {res["accuracy"]}
Precision: {res["precision"]}
Recall: {res["recall"]}
F1: {res["f1"]}
""" + "-"*40 + "\n"
            print(text)
            f.write(text)