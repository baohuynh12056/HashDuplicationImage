from torchvision import models


def preload():
    print("Downloading ResNet50 weights...")
    models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    print("✓ ResNet50 weights cached.")


if __name__ == "__main__":
    preload()
