import torch.nn as nn
import torchvision.models as models


class SimCLRModel(nn.Module):
    def __init__(self, projection_dim=128):
        super().__init__()
        base_model = models.resnet18(weights=None) # ResNet без последнего FC слоя
        num_ftrs = base_model.fc.in_features # Получаем размер выходного вектора энкодера
        base_model.fc = nn.Identity()  
        self.encoder = base_model
        self.projection_head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x) # Получаем представление от энкодера
        z = self.projection_head(h) # Проецируем в пространство меньшей размерности
        return z