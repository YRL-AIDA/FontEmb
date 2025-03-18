
from utils import FontImgGenerator
from PIL import Image
import numpy as np
from model import image_to_gray, ModelDiff, SubCharCNNClassifier
import torch
name_model1 = "model1_loss_29.pt"
name_diff_model = "model_diff_loss_29.pt"

def classifier(model1, model_diff, char_left, char_right):
    gray_image = image_to_gray(char_left, char_right)

    # размерность для баьча
    data_left = torch.Tensor(gray_image[0]).unsqueeze(0).unsqueeze(0)  # (1, 1, 60, 120)
    data_right = torch.Tensor(gray_image[1]).unsqueeze(0).unsqueeze(0)  # (1, 1, 60, 120)

    emb_left = model1(data_left)  # (1, 128)
    emb_right = model1(data_right)  # (1, 128)

    rez = model_diff(emb_left, emb_right) 

    # применяем сигмоидную функцию для получения вероятности
    probability = torch.sigmoid(rez)
    return torch.mean(probability) 
V = []

model1 = SubCharCNNClassifier()
model1.load_state_dict(torch.load(name_model1, weights_only=True))
diff_model = ModelDiff()
diff_model.load_state_dict(torch.load(name_diff_model, weights_only=True))
for i in range(50):
    tmp_img_name = "test_img.png"
    FontImgGenerator().generate_images(name_img=tmp_img_name)
    
    image = Image.open(tmp_img_name)
    # image.show()
    
    image_array = np.array(image)
    
    # разделяем изображение 
    char_left = image_array[:, :120, :]  # левое
    char_right = image_array[:, 120:, :]  # правое
    
    v = classifier(model1, diff_model, char_left, char_right)
    V.append(v.tolist()<0.5)
for i in range(50):
    tmp_img_name = "test_img.png"
    FontImgGenerator().generate_images(name_img=tmp_img_name, same_text=True)
    
    image = Image.open(tmp_img_name)
    # image.show()
    
    image_array = np.array(image)
    
    # разделяем изображение 
    char_left = image_array[:, :120, :]  # левое
    char_right = image_array[:, 120:, :]  # правое
    
    v = classifier(model1, diff_model, char_left, char_right)
    V.append(v.tolist()<0.5)
for i in range(100):
    tmp_img_name = "test_img.png"
    FontImgGenerator().generate_images(name_img=tmp_img_name, style=True)
    
    image = Image.open(tmp_img_name)
    # image.show()
    
    image_array = np.array(image)
    
    # разделяем изображение 
    char_left = image_array[:, :120, :]  # левое
    char_right = image_array[:, 120:, :]  # правое
    
    v = classifier(model1, diff_model, char_left, char_right)
    V.append(v.tolist()>0.5)





print(np.mean(V))