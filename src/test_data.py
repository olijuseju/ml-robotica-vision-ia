from data import get_dataloaders, get_device


def main():
    device = get_device()
    print("Device:", device)

    train_loader, test_loader = get_dataloaders(batch_size=64, num_workers=0)

    images, labels = next(iter(train_loader))

    print("Shape imágenes:", images.shape)  # [64, 3, 32, 32]
    print("Shape labels:", labels.shape)  # [64]
    print("Primeras labels:", labels[:10])

    images = images.to(device)
    labels = labels.to(device)
    print("Device imágenes:", images.device)
    print("Device labels:", labels.device)


if __name__ == "__main__":
    main()
