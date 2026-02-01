import os
from torch.utils.data import DataLoader
from model_architecture.model import SubCharCNNClassifier, ModelDiff
from dataset_class import CharImageDataset
from model_architecture.article_model import ArticleCNNClassifier
from torch import optim
import torch
from torch.nn import BCEWithLogitsLoss
import numpy as np

device = torch.device('cuda:0' if torch.cuda.device_count() != 0 else 'cpu')
LOG_FILE = "log.train.txt"

# def split_index_train_val(dataset, val_split=0.1, shuffle=True, seed=1234, batch_size=64):
#     N = len(dataset)
#     count_batchs = int(N*(1-val_split))//batch_size
#     val_count_batchs = int(N*(val_split))//batch_size
#     train_size = count_batchs * batch_size
#     indexs = [i for i in range(N)]
#     np.random.shuffle(indexs)
#
#
#     train_indexs = indexs[:train_size]
#     val_indexs = indexs[train_size:]
#     batchs_train_indexs = [[train_indexs[k*batch_size+i] for i in range(batch_size)] for k in range(count_batchs)]
#     batch_val_indexs = [[val_indexs[k*batch_size+i] for i in range(batch_size)] for k in range(val_count_batchs)]
#     return batchs_train_indexs, batch_val_indexs
#
#
# def step(model_cnn, model_diff, batch, optimizer, criterion, device, is_train=False):
#     if is_train:
#         optimizer.zero_grad()
#     left_img = torch.cat([b[0][0].unsqueeze(0) for b in batch], dim=0).to(device)
#     right_img = torch.cat([b[0][1].unsqueeze(0) for b in batch], dim=0).to(device)
#     targets = torch.cat([b[1].unsqueeze(0) for b in batch], dim=0).to(device)
#
#     font_emb_left = model_cnn(left_img).to(device)  # (batch_size, 128)
#     font_emb_right = model_cnn(right_img).to(device)  # (batch_size, 128)
#
#     # выход
#     sameness = model_diff(font_emb_left, font_emb_right).to(device)  # (batch_size, 1)
#
#     # Приводим метки к нужной форме
#     targets = targets.view(-1, 1).to(device)
#
#     loss = criterion(sameness, targets)
#     if is_train:
#         loss.backward()
#         optimizer.step()
#     return loss.item()


def log(text):
    with open(LOG_FILE, "a") as f:
        f.write(text+ "\n")

def train(optimizer, criterion,   dataset,
          model_cnn, model_diff, num_epochs, batch_size, name, device):
    log(f"MODEL NAME: {name}" + "="*10)
    MODEL_CNN_FILE = f"{name}_cnn.pt"
    MODEL_DIFF_FILE = f"{name}_diff.pt"

    N = len(dataset)
    N_train = int(0.9 * N)
    N_val = max(1, int(N * 0.1))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [N_train, N_val])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    top_loss = 1
    for epoch in range(num_epochs):
        model_cnn.train()
        model_diff.train()
        train_loss = 0.0

        for (image_pair, label) in train_loader:
            img1, img2 = image_pair
            img1, img2, label = img1.to(device), img2.to(device), label.view(-1, 1).to(device)

            optimizer.zero_grad()
            emb1 = model_cnn(img1)
            emb2 = model_cnn(img2)

            output = model_diff(emb1, emb2)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)


        val_loss = 0.0
        model_cnn.eval()
        model_diff.eval()

        with torch.no_grad():
            for (image_pair, label) in val_loader:
                img1, img2 = image_pair
                img1, img2, label = img1.to(device), img2.to(device), label.view(-1, 1).to(device)

                emb1 = model_cnn(img1)
                emb2 = model_cnn(img2)

                output = model_diff(emb1, emb2)
                loss = criterion(output, label)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if val_loss < top_loss:
            top_loss = val_loss

            if os.path.exists(MODEL_CNN_FILE):
                os.remove(MODEL_CNN_FILE)
            if os.path.exists(MODEL_DIFF_FILE):
                os.remove(MODEL_DIFF_FILE)

            torch.save(model_cnn.state_dict(), MODEL_CNN_FILE)
            torch.save(model_diff.state_dict(), MODEL_DIFF_FILE)
        log(f"Epoch [{epoch + 1}/{num_epochs}], Loss: [train: {train_loss:.4f} / val: {val_loss: .4f} ]")
        print(f"Epoch [{epoch + 1}/{num_epochs}] | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")


if __name__ == "__main__":
    model_cnn = SubCharCNNClassifier().to(device)
    model_diff = ModelDiff().to(device)
    dataset = CharImageDataset('dataset')
    optimizer = optim.Adam(list(model_cnn.parameters()) + list(model_diff.parameters()), lr=0.001)
    criterion = BCEWithLogitsLoss()
    train(optimizer, criterion, dataset, model_cnn, model_diff, num_epochs=30, batch_size=128, name="sub_char", device=device)