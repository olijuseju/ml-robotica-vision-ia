import torch
import torch.nn as nn
import torch.optim as optim

from data import get_dataloaders, get_device
from utils import accuracy_top1, CSVLogger
from models.cnn_minima import CnnMinima


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    epoch: int,
    logger: CSVLogger,
    log_interval: int = 100,
):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = accuracy_top1(logits, targets)

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_correct += acc * batch_size
        running_total += batch_size

        if batch_idx % log_interval == 0:
            print(
                f"[Train] Epoch {epoch} Batch {batch_idx} "
                f"Loss: {loss.item():.4f} Acc: {acc:.4f}"
            )

        logger.log(
            {
                "phase": "train",
                "epoch": epoch,
                "batch": batch_idx,
                "loss": float(loss.item()),
                "acc": float(acc),
            }
        )

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    print(f"[Train] Epoch {epoch} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    epoch: int,
    logger: CSVLogger,
):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)
            acc = accuracy_top1(logits, targets)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            running_correct += acc * batch_size
            running_total += batch_size

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    print(f"[Val]   Epoch {epoch} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    logger.log(
        {
            "phase": "val",
            "epoch": epoch,
            "batch": -1,
            "loss": float(epoch_loss),
            "acc": float(epoch_acc),
        }
    )

    return epoch_loss, epoch_acc


def main():
    device = get_device()
    print("Device:", device)

    train_loader, val_loader = get_dataloaders(batch_size=128, num_workers=0)

    model = CnnMinima(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    logger = CSVLogger(
        filepath="runs/cnn_minima/metrics.csv",
        fieldnames=["phase", "epoch", "batch", "loss", "acc"],
    )

    num_epochs = 5

    best_val_acc = 0.0
    for epoch in range(num_epochs):
        train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger
        )
        _, val_acc = evaluate(model, val_loader, criterion, device, epoch, logger)

        # Guardar el mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                },
                "runs/cnn_minima/best.pt",
            )
            print(f"Nuevo mejor modelo con val_acc = {val_acc:.4f}")


if __name__ == "__main__":
    main()
