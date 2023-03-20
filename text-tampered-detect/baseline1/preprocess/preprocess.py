# %%
# load pth
tempered_data_path = r'/home/wuziyang/Projects/data/text_manipulation_detection/train/train/tampered/imgs'
untampered_data_path = r'/home/wuziyang/Projects/data/text_manipulation_detection/train/train/untampered'
dataset_path = r'/home/wuziyang/Projects/data/text_manipulation_detection/dataset'

import os

temper_dict = {} 
untamper_dict = {}
tamper_path_list = []
untamper_path_list = []
temper_list = os.listdir(tempered_data_path)
untamper_list = os.listdir(untampered_data_path)
for p in temper_list:
    p = os.path.join(tempered_data_path, p)
    temper_dict[p] = 1
    tamper_path_list.append(p)
for p in untamper_list:
    p = os.path.join(untampered_data_path, p)
    untamper_dict[p] = 0
    untamper_path_list.append(p)

# %%
# prepare data
import random
import numpy as np
import cv2 as cv
from PIL import Image
from transform_tamplate import *

train_tamper_list = random.sample(tamper_path_list, int(0.8 * len(tamper_path_list)))
train_untamper_list = random.sample(untamper_path_list, int(0.8 * len(untamper_path_list)))
test_tamper_list = [i for i in tamper_path_list if i not in train_tamper_list]
test_untamper_list = [i for i in untamper_path_list if i not in train_untamper_list]

np.save(os.path.join(dataset_path, 'test/0/data.npy'), test_tamper_list)
np.save(os.path.join(dataset_path, 'test/1/data.npy'), test_untamper_list)

print('test data ready')

# # 数据增强
# tamper_dataset_path = os.path.join(dataset_path, 'tamper')
# for i in train_tamper_list:
#     img = Image.open(i)
#     img.save(os.path.join(tamper_dataset_path, i))
#     i1 = data_aug1(i)
#     i1.save(os.path.join(tamper_dataset_path, i.split('/')[-1].split('.')[0] + '1.png'))
#     i2 = data_aug2(i)
#     i2.save(os.path.join(tamper_dataset_path, i.split('/')[-1].split('.')[0] + '2.png'))
#     i3 = data_aug3(i)
#     i3.save(os.path.join(tamper_dataset_path, i.split('/')[-1].split('.')[0] + '3.png'))

print('train tamper data ready')
untamper_dataset_path = os.path.join(dataset_path, 'untamper')
for i in train_untamper_list:
    img = Image.open(i)
    img.save(os.path.join(untamper_dataset_path, i))
    i1 = data_aug1(i)
    i1.save(os.path.join(untamper_dataset_path, i.split('/')[-1].split('.')[0] + '1.png'))
    i2 = data_aug2(i)
    i2.save(os.path.join(untamper_dataset_path, i.split('/')[-1].split('.')[0] + '2.png'))
    i3 = data_aug3(i)
    i3.save(os.path.join(untamper_dataset_path, i.split('/')[-1].split('.')[0] + '3.png'))

print('train untamper data ready')

# %%
# train
# from sklearn.model_selection import StratifiedShuffleSplit
# import torchvision

# resnet = torchvision.models.resnet101(pretrained=True)