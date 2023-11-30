# Transforms
from torchvision.transforms import v2

train_transforms = v2.Compose([
    v2.ColorJitter(brightness=1.0, contrast=0.5, saturation=1, hue=0.1),
    v2.RandomEqualize(0.4),
    v2.AugMix(),
    v2.RandomHorizontalFlip(p=0.3),
    v2.RandomVerticalFlip(0.3),
    v2.GaussianBlur((3,3)),
    v2.RandomRotation(30),
    v2.Resize(size=(50,50)),
    v2.ToImageTensor(),
])
validation_transforms =  v2.Compose([
    v2.Resize(size=(50,50)),
    v2.ToImageTensor(),
])
transforms = v2.Compose([
    v2.Resize(size=(50, 50)),
    v2.ToImageTensor(),
])