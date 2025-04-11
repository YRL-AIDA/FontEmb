import os
from model_architecture.model import SubCharCNNClassifier, ModelDiff
from dataset_class import CharImageDataset, MultiDirCharImageDataset
from model_architecture.article_model import ArticleCNNClassifier
from torch import optim
import torch
from torch.nn import BCEWithLogitsLoss
import numpy as np

device = torch.device('cuda:0' if torch.cuda.device_count() != 0 else 'cpu')
LOG_FILE = "log.train.txt"

def split_index_train_val(dataset, val_split=0.1, shuffle=True, seed=1234, batch_size=64):
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


def step(model_cnn, model_diff, batch, optimizer, criterion, device, is_train=False):
    if is_train:
        optimizer.zero_grad()
    left_img = torch.cat([b[0][0].unsqueeze(0) for b in batch], dim=0).to(device)
    right_img = torch.cat([b[0][1].unsqueeze(0) for b in batch], dim=0).to(device)
    targets = torch.cat([b[1].unsqueeze(0) for b in batch], dim=0).to(device)
    
    font_emb_left = model_cnn(left_img).to(device)  # (batch_size, 128)
    font_emb_right = model_cnn(right_img).to(device)  # (batch_size, 128)

    # выход 
    sameness = model_diff(font_emb_left, font_emb_right).to(device)  # (batch_size, 1)

    # Приводим метки к нужной форме
    targets = targets.view(-1, 1).to(device)  

    loss = criterion(sameness, targets)
    if is_train:
        loss.backward()
        optimizer.step()
    return loss.item()


def log(text):
    with open(LOG_FILE, "a") as f:
        f.write(text+ "\n")

def train(optimizer, criterion,   dataset, 
          model_cnn, model_diff, num_epochs, batch_size, name, device):
    log(f"MODEL NAME: {name}" + "="*10)
    MODEL_CNN_FILE = f"{name}_cnn.pt"
    MODEL_DIFF_FILE = f"{name}_diff.pt"
    train_index, val_index = split_index_train_val(dataset, batch_size=batch_size, shuffle=True)
    
    top_loss = 1
    for epoch in range(num_epochs):
        model_cnn.train()
        model_diff.train()
        running_loss = 0.0

        for i, (batch_index) in enumerate(train_index):
            batch= [dataset[j] for j in batch_index]
            train_loss = step(model_cnn, model_diff, batch, optimizer, criterion, device, is_train=True)
            running_loss += train_loss
            if i % 10 == 9:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_index)}], Loss: {train_loss:.4f}", end = '\r')

        val_loss = 0.0
        for i, (batch_index) in enumerate(val_index):
            batch = [dataset[j] for j in batch_index]
            val_loss += step(model_cnn, model_diff, batch, optimizer, criterion, device, is_train=False)
        val_loss = val_loss / len(val_index)

        if val_loss < top_loss:
            top_loss = val_loss

            if os.path.exists(MODEL_CNN_FILE):
                os.remove(MODEL_CNN_FILE)
            if os.path.exists(MODEL_DIFF_FILE):
                os.remove(MODEL_DIFF_FILE)

            torch.save(model_cnn.state_dict(), MODEL_CNN_FILE)
            torch.save(model_diff.state_dict(), MODEL_DIFF_FILE)
        log(f"Epoch [{epoch + 1}/{num_epochs}], Loss: [train: {running_loss/len(train_index):.4f} / val: {val_loss: .4f} ]")


if __name__ == "__main__":
    model_cnn = SubCharCNNClassifier().to(device)
    model_diff = ModelDiff().to(device)
    dataset = CharImageDataset('train_dataset')
    optimizer = optim.Adam(list(model_cnn.parameters()) + list(model_diff.parameters()), lr=0.001)
    criterion = BCEWithLogitsLoss()
    train(optimizer, criterion, dataset, model_cnn, model_diff, num_epochs=20, batch_size=8, name="sub_char", device=device)