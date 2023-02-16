import sys
# sys.path.append('../')
# from preprocess.loadImage import LoadImage
from loadImage import LoadImage
from torchvision import transforms

data_aug1 = transforms.Compose([
    LoadImage(),
    transforms.RandomHorizontalFlip(p=1),
])

data_aug2 = transforms.Compose([
    LoadImage(),
    transforms.RandomHorizontalFlip(p=1),
    transforms.RandomVerticalFlip(p=1),
])

data_aug3 = transforms.Compose([
    LoadImage(),
    transforms.RandomVerticalFlip(p=1),
])

temper_transform = transforms.Compose([
    LoadImage(),
    transforms.Resize(384),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

untemper_transform = transforms.Compose([
    LoadImage(),
    transforms.Resize(512),
    transforms.RandomResizedCrop(384),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

