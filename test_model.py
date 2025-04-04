import numpy as np
from PIL import Image
from model import image_to_gray, ModelDiff, SubCharCNNClassifier
from article_model import ArticleCNNClassifier
import torch
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


name_model1 = "article_model2.pt"
name_diff_model = "article_model_diff2.pt"

# model1 = SubCharCNNClassifier()
model1 = ArticleCNNClassifier()
model1.load_state_dict(torch.load(name_model1, map_location=torch.device('cpu')))
model1.eval()

diff_model = ModelDiff()
diff_model.load_state_dict(torch.load(name_diff_model, map_location=torch.device('cpu')))
diff_model.eval()


def classifier(model1, model_diff, char_left, char_right):
    gray_image = image_to_gray(char_left, char_right)
    data_left = torch.Tensor(gray_image[0]).unsqueeze(0).unsqueeze(0)
    data_right = torch.Tensor(gray_image[1]).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        emb_left = model1(data_left)
        emb_right = model1(data_right)
        rez = model_diff(emb_left, emb_right)
        probability = torch.sigmoid(rez)
    return probability.item()


def testing(test_dir, test_name):
    class_dirs = ['0', '1']
    results = {0: [], 1: []}

    for class_idx, class_dir in enumerate(class_dirs):
        class_path = os.path.join(test_dir, class_dir)
        for img_name in os.listdir(class_path):
            if img_name.endswith('.png'):
                image_path = os.path.join(class_path, img_name)
                image = Image.open(image_path)
                image_array = np.array(image)

                char_left = image_array[:, :40, :]
                char_right = image_array[:, 40:, :]

                prob = classifier(model1, diff_model, char_left, char_right)
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

    report = f"""
    ======================
    Датасет: {test_name}
    Модель 1: {name_model1}
    Модель Diff: {name_diff_model}

    Результаты:
    Класс 0 (разные шрифты): {results[0]}
    Класс 1 (одинаковые шрифты): {results[1]}

    Метрики:
    Accuracy: {accuracy:.4f}
    Precision: {precision:.4f}
    Recall: {recall:.4f}
    F1: {f1:.4f}
    """

    print(report)
    return report


test_dirs = ['test', 'test1', 'test2']
results_file = "test_results.txt"

with open(results_file, 'a', encoding='utf-8') as f:
    for test_dir in test_dirs:
        report = testing(test_dir, test_dir)
        f.write(report)