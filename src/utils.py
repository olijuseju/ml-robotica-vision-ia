from pathlib import Path
from typing import Dict, Any, List

import csv
import torch


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calcula la accuracy top-1 en un batch.
    logits: [B, num_classes]
    targets: [B]
    """
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        correct = (preds == targets).float().sum().item()
        total = targets.numel()
    return correct / total


class CSVLogger:
    """
    Logger sencillo a CSV para métricas por batch/epoch.
    """

    def __init__(self, filepath: str, fieldnames: List[str]):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames

        if not self.filepath.exists():
            with self.filepath.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        with self.filepath.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
