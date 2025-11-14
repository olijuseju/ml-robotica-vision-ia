import torch
from models.cnn_minima import CnnMinima


def main():
    x = torch.randn(1, 3, 32, 32)  # imagen dummy
    model = CnnMinima()

    print("Entrada:", x.shape)

    x = model.block1(x)
    print("Tras block1:", x.shape)

    x = model.block2(x)
    print("Tras block2:", x.shape)

    x = model.avgpool(x)
    print("Tras avgpool:", x.shape)

    x = x.view(x.size(0), -1)
    print("Tras flatten:", x.shape)

    x = model.fc(x)
    print("Tras fc:", x.shape)


main()
