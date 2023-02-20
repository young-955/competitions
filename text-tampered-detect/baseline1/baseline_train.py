# %%
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
import numpy as np
import torchvision
import cv2
import random
import copy
from torch.autograd import Variable
import sys
import pickle
import math
from torchvision import transforms as T
from dataset import tamper_Dataset
from torch.utils.data import DataLoader
from loadImage import LoadImage

batch_size = 1
epoch_num = 5

train_path = r'/home/wuziyang/Projects/data/text_manipulation_detection/dataset'
train_path = r'C:\Users\young\Documents\pywork\pytorch-learn\competitions\data\text_manipulation_detection\train'
# val_path = r'/home/wuziyang/Projects/data/text_manipulation_detection/dataset/test'

# make dataset
tamper_path = os.path.join(train_path, 'tampered/imgs')
untamper_path = os.path.join(train_path, 'untampered')

tamper_list = [os.path.join(tamper_path, p) for p in os.listdir(tamper_path)]
untamper_list = [os.path.join(untamper_path, p) for p in os.listdir(untamper_path)]
train_dataset = tamper_Dataset(tamper_list, untamper_list)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

# val_tp_data = np.load(os.path.join(val_path, '0/data.npy'))
# val_utp_data = np.load(os.path.join(val_path, '1/data.npy'))
# val_dataset = tamper_Dataset(val_tp_data, val_utp_data)
# val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

res_all = {}

# prepare model
import timm
model = timm.create_model('tf_efficientnet_b1', pretrained=True, num_classes=2)
# model = torch.load('pretrain_baseline.pth',map_location=torch.device('cpu'))
model = model.cuda()
# model.eval()

criterion = nn.CrossEntropyLoss().cuda()
optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

trans = {384:T.Compose([T.Resize(384), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
         128:T.Compose([T.ToTensor(),T.RandomCrop((128, 128))]),
         224:T.Compose([T.ToTensor(),T.Resize((224,224))]),
         299:T.Compose([T.ToTensor(),T.Resize((299,299)),T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
         256:T.Compose([T.ToTensor(), T.Resize((256, 256)),T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
         1280:T.Compose([T.ToTensor(),T.Resize((128,128))]),
         0:T.Compose([T.ToTensor(), ])}

trans = trans[384]

# train model
gstep = 0
grecall = 0
for epoch in range(epoch_num):
    for step, item in enumerate(train_loader):
        batch_image, batch_label = item
        pred = model(batch_image)
        batch_label = torch.nn.functional.one_hot(batch_label.type(torch.long), 2).type(torch.float).cuda()
        loss = criterion(pred, batch_label)

        optim.zero_grad()
        loss.backward()

        optim.step()

        gstep += 1
        print(gstep)
        if gstep % 500 == 0:
            recall = eval(model)
            print(f'step {gstep}, recall {recall}')
            if recall > grecall:
                grecall = recall
                torch.save(model, f'model_{recall}.pth')


# trans = trans[384]
# def detect(img_path):
#     img = cv2.imread(img_path)
#     img = trans(img)
#     img = img.unsqueeze(0).float()

#     img = img.cuda()

#     img = img.to(torch.float32)

#     res = model(img)
#     prob = torch.nn.functional.softmax(res)
#     return prob[:,1].detach().cpu().numpy()

# test_path = '../data/text_manipulation_detection/test/imgs'
# with open(f"submission.txt", "w") as f:
#     for p in os.listdir(test_path):
#         rela_path = os.path.join(test_path, p)
#         print(rela_path)
#         res = detect(rela_path)
#         f.write(f"{p} {res[0]}\n")
#         f.flush()

