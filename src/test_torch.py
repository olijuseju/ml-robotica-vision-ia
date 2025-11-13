import torch
import torch.nn as nn


def main():
    # 1) Tensor de prueba
    x = torch.randn(4, 3, 32, 32)  # batch=4, canales=3, 32x32

    # 2) Capa convolucional
    conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)

    # 3) Pasamos datos por la capa
    y = conv(x)

    print("Entrada:", x.shape)
    print("Salida:", y.shape)

    # 4) Comprobamos autograd
    loss = y.mean()
    loss.backward()

    # Miramos el gradiente de los pesos de la conv
    print("Gradiente de conv.weight:", conv.weight.grad.shape)


if __name__ == "__main__":
    main()
