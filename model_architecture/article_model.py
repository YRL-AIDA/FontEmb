import torch.nn as nn

class ArticleCNNClassifier(nn.Module):
    def __init__(self):
        super(ArticleCNNClassifier, self).__init__()

        # сверточные слои
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        # максимальный пулинг
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # полносвязный слой
        self.fc1 = nn.Linear(32 * 10 * 10, 256)
        self.fc2 = nn.Linear(256, 8)

        # функция активации
        self.relu = nn.ReLU()

    def forward(self, x):
        # применяем свертки и пулинг
        x = self.relu(self.conv1(x))  # (batch_size, 16, 40, 40)
        x = self.relu(self.conv2(x))  # (batch_size,  16, 40, 40)

        x = self.pool(x)

        x = self.relu(self.conv3(x))  # (batch_size, 32, 20, 20)
        x = self.relu(self.conv4(x))  # (batch_size, 32, 20, 20)

        x = self.pool(x)

        # делаем вектор одномерным для fc1
        x = x.view(x.size(0), -1)  # (batch_size, 32 * 10 * 10)

        # применяем полносвязный слой и relu
        x = self.relu(self.fc1(x))  # (batch_size, 64)
        x = self.relu(self.fc2(x))

        return x