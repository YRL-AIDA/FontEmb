from model import SubCharCNNClassifier, ModelDiff, CharImageDataset
from torch.utils.data import DataLoader
from torch import optim
import torch
from torch.nn import BCEWithLogitsLoss
import numpy as np

device = torch.device('cuda:0' if torch.cuda.device_count() != 0 else 'cpu')

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
    left_img = torch.cat([b[0][0].unsqueeze(0) for b in batch], dim=0).to(device)
    right_img = torch.cat([b[0][1].unsqueeze(0) for b in batch], dim=0).to(device)
    targets = torch.cat([b[1].unsqueeze(0) for b in batch], dim=0).to(device)
    
    font_emb_left = model1(left_img).to(device)  # (batch_size, 128)
    font_emb_right = model1(right_img).to(device)  # (batch_size, 128)

    # выход 
    sameness = model_diff(font_emb_left, font_emb_right).to(device)  # (batch_size, 1)

    # Приводим метки к нужной форме
    targets = targets.view(-1, 1) .to(device) 

    loss = criterion(sameness, targets).to(device)
    return loss.item()

def train_step(batch):
    optimizer.zero_grad()
    left_img = torch.cat([b[0][0].unsqueeze(0) for b in batch], dim=0).to(device)
    right_img = torch.cat([b[0][1].unsqueeze(0) for b in batch], dim=0).to(device)
    targets = torch.cat([b[1].unsqueeze(0) for b in batch], dim=0).to(device)
    
    font_emb_left = model1(left_img).to(device)  # (batch_size, 128)
    font_emb_right = model1(right_img).to(device)  # (batch_size, 128)

    # выход 
    sameness = model_diff(font_emb_left, font_emb_right).to(device)  # (batch_size, 1)

    # Приводим метки к нужной форме
    targets = targets.view(-1, 1).to(device)  

    loss = criterion(sameness, targets)
    loss.backward()
    optimizer.step()
    return loss.item()


LOG_FILE = "log.train.txt"
num_epochs = 20

model1 = SubCharCNNClassifier().to(device)
# model2 = SubCharCNNClassifier()
model_diff = ModelDiff().to(device)


criterion = BCEWithLogitsLoss().to(device) 
# optimizer = optim.Adam(list(model1.parameters()) + list(model2.parameters()) + list(model_diff.parameters()), lr=0.001)
optimizer = optim.Adam(list(model1.parameters()) + list(model_diff.parameters()), lr=0.001)

dataset = CharImageDataset("dataset/") 
train_index, val_index = split_index_train_val(dataset, batch_size=256, shuffle=True)


with open(LOG_FILE, "w") as f:
    f.write("START_LEANING\n")

top_loss = 1

for epoch in range(num_epochs):
    model1.train()
    model_diff.train()
    
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
    val_loss = val_loss/len(val_index)
    if val_loss < top_loss:
        top_loss = val_loss
        name_loss = f"loss_{int(top_loss*100)}"
        torch.save(model1.state_dict(), f'model1_{name_loss}.pt')
        torch.save(model_diff.state_dict(), f'model_diff_{name_loss}.pt')

    with open(LOG_FILE, "a") as f:
        f.write(f"Epoch [{epoch + 1}/{num_epochs}], Loss: [train: {running_loss/len(train_index):.4f} / val: {val_loss: .4f} ] \n")