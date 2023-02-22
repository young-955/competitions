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
from eval import eval
from loss.focal_loss import Focal_Loss
import timm

batch_size = 16
epoch_num = 3
eval_step = 500
label_weight = 10

train_path = r'/home/wuziyang/Projects/data/text_manipulation_detection/dataset'
# train_path = r'C:\Users\young\Documents\pywork\pytorch-learn\competitions\data\text_manipulation_detection\train'
# val_path = r'/home/wuziyang/Projects/data/text_manipulation_detection/dataset/test'

# make dataset
tamper_path = os.path.join(train_path, 'tamper')
untamper_path = os.path.join(train_path, 'untamper')

tamper_list = [os.path.join(tamper_path, p) for p in os.listdir(tamper_path)]
untamper_list = [os.path.join(untamper_path, p) for p in os.listdir(untamper_path)]
train_tamper_dataset = tamper_Dataset(tamper_list)
train_tamper_loader = DataLoader(dataset=train_tamper_dataset, batch_size=batch_size, shuffle=False)
train_untamper_dataset = tamper_Dataset(untamper_list)
train_untamper_loader = DataLoader(dataset=train_untamper_dataset, batch_size=batch_size, shuffle=False)

# val_tp_data = np.load(os.path.join(val_path, '0/data.npy'))
# val_utp_data = np.load(os.path.join(val_path, '1/data.npy'))
# val_dataset = tamper_Dataset(val_tp_data, val_utp_data)
# val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

res_all = {}

# prepare model
tamper_model = timm.create_model('tf_efficientnet_b7', pretrained=True, num_classes=2)
untamper_model = timm.create_model('tf_efficientnet_b7', pretrained=True, num_classes=2)
# 调整finetune层
for name, param in tamper_model.named_parameters():
    if 'bn' in name or 'classifier' in name:
        print(name)
        param.requires_grad = True
    else:
        param.requires_grad = False
for name, param in untamper_model.named_parameters():
    if 'bn' in name or 'classifier' in name:
        print(name)
        param.requires_grad = True
    else:
        param.requires_grad = False

# model = torch.load('pretrain_baseline.pth',map_location=torch.device('cpu'))
tamper_model = tamper_model.cuda()
tamper_model.train()
untamper_model = untamper_model.cuda()
untamper_model.train()
# recall = eval(model)
# print(f'recall {recall}')
# model.eval()

tamper_criterion = nn.CrossEntropyLoss().cuda()
untamper_criterion = nn.CrossEntropyLoss().cuda()
# criterion = Focal_Loss()
# optim = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
# filter(lambda p: p.requires_grad, model.parameters())
tamper_optim = torch.optim.SGD(filter(lambda p: p.requires_grad, tamper_model.parameters()), lr=0.0001, momentum=0.9, weight_decay=0.01)
untamper_optim = torch.optim.SGD(filter(lambda p: p.requires_grad, untamper_model.parameters()), lr=0.0001, momentum=0.9, weight_decay=0.01)

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
# hard_example = {}
gloss = 0
for epoch in range(epoch_num):
    for step, item in enumerate(train_tamper_loader):
        batch_image, batch_label, batch_imgname = item
        # if batch_label == 0:
        pred = tamper_model(batch_image)
        batch_label = batch_label.type(torch.float).cuda()
        batch_label = torch.nn.functional.one_hot(batch_label.type(torch.long), 2).type(torch.float).cuda()

        loss = tamper_criterion(pred, batch_label)
        gloss += loss.detach().cpu()
        # if batch_label[:, 1] == 0:
        #     loss *= label_weight
        # hard_example[loss] = batch_imgname

        tamper_optim.zero_grad()
        loss.backward()

        tamper_optim.step()

        gstep += 1
        # print(gstep)
        if gstep % eval_step == 0:
            recall = eval(tamper_model)
            gloss = gloss / eval_step
            print(f'step {gstep}, recall {recall}, aver loss: {gloss}')
            if recall > grecall:
                grecall = recall
                # if recall > 0.3:
                torch.save(tamper_model, f'model_tamper_{recall}.pth')
    # hard_example.keys().sort
torch.save(tamper_model, f'model_tamper_{recall}.pth')

gstep = 0
grecall = 0
gloss = 0
for epoch in range(epoch_num):
    for step, item in enumerate(train_untamper_loader):
        batch_image, batch_label, batch_imgname = item
        pred = untamper_model(batch_image)
        batch_label = batch_label.type(torch.float).cuda()
        batch_label = torch.nn.functional.one_hot(batch_label.type(torch.long), 2).type(torch.float).cuda()

        loss = untamper_criterion(pred, batch_label)
        gloss += loss.detach().cpu()

        untamper_optim.zero_grad()
        loss.backward()

        tamper_optim.step()

        gstep += 1
        if gstep % eval_step == 0:
            recall = eval(untamper_model)
            gloss = gloss / eval_step
            print(f'step {gstep}, recall {recall}, aver loss: {gloss}')
            if recall > grecall:
                grecall = recall
                torch.save(untamper_model, f'model_untamper_{recall}.pth')
torch.save(untamper_model, f'model_untamper_{recall}.pth')

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

