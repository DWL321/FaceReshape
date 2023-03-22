import torchvision.transforms as transforms

imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalize = transforms.Compose(
    [
        transforms.ToTensor(),
        imagenet_norm,
    ]
)
