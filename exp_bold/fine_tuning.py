import sys, os
sys.path.append("..")
from model_architecture.model import SubCharCNNClassifier
import torch
import numpy as np
from torch.nn import BCEWithLogitsLoss, Module, Linear, ReLU
from torch import optim
from torch.utils.data  import Dataset
from PIL import Image
import cv2

device = torch.device('cuda:0' if torch.cuda.device_count() != 0 else 'cpu')
device = torch.device('cpu')

class BoldTask(Module):
    def __init__(self):
        super(BoldTask, self).__init__()
        self.fc1 = Linear(8, 8)  
        self.fc2 = Linear(8, 1)  
        self.relu = ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))  # (batch_size, 8)
        x = self.fc2(x)  # (batch_size, 1)
        return x


def label_to_vec(text):
    return torch.Tensor([float(text[0])])


def image_to_gray(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0
    return np.array(grayscale_image)

class CharImageDataset(Dataset):
    def __init__(self, img_dir, transform=image_to_gray, target_transform=label_to_vec):
        self.img_dir = img_dir
        self.labels = ['0', '1']
        self.counts = [len(os.listdir(os.path.join(self.img_dir, label))) for label in self.labels]
        self.count = sum(self.counts)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        label, i = self.__get_label_and_i_from_idx(idx)
        img_path = os.path.join(self.img_dir, label, f"image_{i}.png")
        image = np.array(Image.open(img_path))
        if  self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return torch.Tensor(image).unsqueeze(0), label

    def __get_label_and_i_from_idx(self, idx):
        k = 0
        while (idx - self.counts[k]) >= 0:
            idx -= self.counts[k]
            k += 1
        return self.labels[k], idx

def split_index_train_val(dataset, val_split=0.1, shuffle=True, seed=1234,batch_size=64):
    N = len(dataset)
    count_batchs = int(N*(1-val_split))//batch_size
    val_count_batchs = int(N*(val_split))//batch_size
    train_size = count_batchs * batch_size 
    indexs = [i for i in range(N)]
    np.random.shuffle(indexs)
    train_indexs = indexs[:train_size]
    val_indexs = indexs[train_size:]
    batchs_train_indexs = [[train_indexs[k*batch_size+i] for i in range(batch_size)] for k in range(count_batchs)]
    batch_val_indexs = [[val_indexs[k*batch_size+i] for i in range(batch_size)] for k in range(val_count_batchs)]
    return batchs_train_indexs, batch_val_indexs   


def validation(batch):
    imgs = torch.cat([b[0].unsqueeze(0) for b in batch], dim=0).to(device)
    targets = torch.cat([b[1].unsqueeze(0) for b in batch], dim=0).to(device)
    
    font_emb_img = model1(imgs).to(device)  # (batch_size, 128)

    # выход 
    bold = model_bold(font_emb_img).to(device)  # (batch_size, 1)

    # Приводим метки к нужной форме
    targets = targets.view(-1, 1).to(device)  

    loss = criterion(bold, targets)
    return loss.item()


def train_step(batch):
    optimizer.zero_grad()
    imgs = torch.cat([b[0].unsqueeze(0) for b in batch], dim=0).to(device)
    targets = torch.cat([b[1].unsqueeze(0) for b in batch], dim=0).to(device)
    
    font_emb_img = model1(imgs).to(device)  # (batch_size, 128)

    # выход 
    bold = model_bold(font_emb_img).to(device)  # (batch_size, 1)

    # Приводим метки к нужной форме
    targets = targets.view(-1, 1).to(device)  

    loss = criterion(bold, targets)
    loss.backward()
    optimizer.step()
    return loss.item()

if __name__ == "__main__":
    LOG_FILE = "log.train.txt"
    MODEL_BOLD_FILE = "model_bold.pt"
    num_epochs = 100

    model1 = SubCharCNNClassifier().to(device)
    model1.load_state_dict(torch.load(os.path.join("..", "model1.pt"), map_location=torch.device('cpu')))
    model1.eval()
    # model2 = SubCharCNNClassifier()
    model_bold = BoldTask().to(device)


    criterion = BCEWithLogitsLoss().to(device) 
    # optimizer = optim.Adam(list(model1.parameters()) + list(model2.parameters()) + list(model_diff.parameters()), lr=0.001)
    optimizer = optim.Adam(list(model_bold.parameters()), lr=0.0025)

    dataset = CharImageDataset("dataset/")
    train_index, val_index = split_index_train_val(dataset, batch_size=32, shuffle=True)


    with open(LOG_FILE, "w") as f:
        f.write("START_LEANING\n")

    top_loss = 1

    for epoch in range(num_epochs):
        model_bold.train()
        running_loss = 0.0

        for i, (batch_index) in enumerate(train_index):
            batch= [dataset[j] for j in batch_index]
            train_loss = train_step(batch)
            running_loss += train_loss

            if i % 10 == 9:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_index)}], Loss: {train_loss:.4f}", end = '\r')

        val_loss = 0.0
        for i, (batch_index) in enumerate(val_index):
            val_loss += validation([dataset[j] for j in batch_index])
        val_loss = val_loss / len(val_index)

        if val_loss < top_loss:
            top_loss = val_loss

            if os.path.exists(MODEL_BOLD_FILE):
                os.remove(MODEL_BOLD_FILE)

            torch.save(model_bold.state_dict(), MODEL_BOLD_FILE)

        with open(LOG_FILE, "a") as f:
            f.write(f"Epoch [{epoch + 1}/{num_epochs}], Loss: [train: {running_loss/len(train_index):.4f} / val: {val_loss: .4f} ] \n")