import torch
import torch.nn as nn


class CnnMinima(nn.Module):
    """
    CNN sencilla para CIFAR-10.
    Entrada: [B, 3, 32, 32]
    Salida: [B, 10]
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Bloque 1: 3 -> 32 canales
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # [B, 32, 32, 32]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # [B, 32, 32, 32]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # [B, 32, 16, 16]
        )

        # Bloque 2: 32 -> 64 canales
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B, 64, 16, 16]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # [B, 64, 16, 16]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # [B, 64, 8, 8]
        )

        # Cabeza de clasificación
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # [B, 64, 1, 1]
        self.fc = nn.Linear(64, num_classes)  # [B, 10]

        # (Opcional) inicialización básica
        self._init_weights()

    def _init_weights(self):
        # Inicializamos las convoluciones de forma razonable
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 32, 32]
        x = self.block1(x)  # [B, 32, 16, 16]
        x = self.block2(x)  # [B, 64, 8, 8]
        x = self.avgpool(x)  # [B, 64, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 64]
        x = self.fc(x)  # [B, 10]
        return x
