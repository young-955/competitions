# %%
# load pth
tempered_data_path = r'C:\Users\young\Documents\pywork\pytorch-learn\competitions\data\text_manipulation_detection\train\tampered\imgs'
untampered_data_path = r'C:\Users\young\Documents\pywork\pytorch-learn\competitions\data\text_manipulation_detection\train\untampered'
dataset_path = r'C:\Users\young\Documents\pywork\pytorch-learn\competitions\data\text_manipulation_detection\dataset'

import os

temper_dict = {} 
untamper_dict = {}
tamper_path_list = []
untamper_path_list = []
temper_list = os.listdir(tempered_data_path)
untamper_list = os.listdir(untampered_data_path)
for p in temper_list:
    p = os.path.join(tempered_data_path, p)
    temper_dict[p] = 0
    tamper_path_list.append(p)
for p in untamper_list:
    p = os.path.join(untampered_data_path, p)
    untamper_dict[p] = 1
    untamper_path_list.append(p)

# %%
# prepare data
from torchvision import transforms
from loadImage import LoadImage
import random
import numpy as np

train_tamper_list = random.sample(tamper_path_list, int(0.8 * len(tamper_path_list)))
train_untamper_list = random.sample(untamper_list, int(0.8 * len(untamper_list)))
test_tamper_list = [i for i in tamper_path_list if i not in train_tamper_list]
test_untamper_list = [i for i in untamper_list if i not in train_untamper_list]

np.save(os.path.join(dataset_path, 'test/0/data.npy'), test_tamper_list)
np.save(os.path.join(dataset_path, 'test/1/data.npy'), test_untamper_list)

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
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

untemper_transform = transforms.Compose([
    LoadImage(),
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# %%
# train
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision

resnet = torchvision.models.resnet101(pretrained=True)