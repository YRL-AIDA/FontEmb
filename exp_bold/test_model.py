import sys, os
sys.path.append("..")
import numpy as np
from PIL import Image
from model_architecture.model import  SubCharCNNClassifier
import torch
from fine_tuning import BoldTask, image_to_gray
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


name_model1 = os.path.join("..", "model1.pt")
name_bold_model = "model_bold.pt"

model1 = SubCharCNNClassifier()
model1.load_state_dict(torch.load(name_model1, map_location=torch.device('cpu')))
model1.eval()

bold_model = BoldTask()
bold_model.load_state_dict(torch.load(name_bold_model, map_location=torch.device('cpu')))
bold_model.eval()


def classifier(model1, bold_model, char):
    gray_image = image_to_gray(char)
    char_img = torch.Tensor(gray_image).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        emb_img = model1(char_img)
        rez = bold_model(emb_img)
        probability = torch.sigmoid(rez)
    return probability.item()

# Выбираем данные для теста
test_dir = 'test'
class_dirs = ['0', '1']
results = {0: [], 1: []}

for class_idx, class_dir in enumerate(class_dirs):
    class_path = os.path.join(test_dir, class_dir)
    for img_name in os.listdir(class_path):
        if img_name.endswith('.png'):
            image_path = os.path.join(class_path, img_name)
            image = Image.open(image_path)
            image_array = np.array(image)
            prob = classifier(model1, bold_model, image_array)
            print(prob)
            pred = 1 if prob > 0.5 else 0
            results[class_idx].append(pred)


true_labels = []
predictions = []

for class_idx in [0, 1]:
    true_labels.extend([class_idx] * len(results[class_idx]))
    predictions.extend(results[class_idx])


accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)


results_file = "test_results.txt"

report = f"""
======================
Модель 1: {name_model1}
Модель Bold: {name_bold_model}

Результаты:
Класс 0 (обычные): {results[0]}
Класс 1 (жирные): {results[1]}

Метрики:
Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1: {f1:.4f}
"""

print(report)
with open(results_file, 'a', encoding='utf-8') as f:
    f.write(report)

