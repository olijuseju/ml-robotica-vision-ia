import torch
import torch.nn as nn
import torch.optim as optim

from data import get_dataloaders, get_device
from utils import accuracy_top1, CSVLogger


class TinyNet(nn.Module):
    """
    Red mínima para probar el pipeline.
    No es la CNN 'buena', solo un juguete.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def main():
    device = get_device()
    print("Device:", device)

    train_loader, _ = get_dataloaders(batch_size=64, num_workers=0)

    model = TinyNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    logger = CSVLogger(
        filepath="runs/test_loop/metrics.csv",
        fieldnames=["epoch", "batch", "loss", "acc"],
    )

    model.train()
    epoch = 0
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = accuracy_top1(logits, targets)

        if batch_idx % 50 == 0:
            print(
                f"Epoch {epoch} Batch {batch_idx} "
                f"Loss: {loss.item():.4f} Acc: {acc:.4f}"
            )

        logger.log(
            {
                "epoch": epoch,
                "batch": batch_idx,
                "loss": float(loss.item()),
                "acc": float(acc),
            }
        )

        if batch_idx >= 100:
            break


if __name__ == "__main__":
    main()
