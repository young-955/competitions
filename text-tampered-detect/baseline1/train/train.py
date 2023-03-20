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
from dataset import *
from torch.utils.data import DataLoader
from loadImage import LoadImage
from eval import eval
from loss.focal_loss import Focal_Loss
import torch.nn.functional as F
import timm
import argparse
# from loss.ohem_loss import ohem_loss

'''
    'resnet101': ['layer4', 'fc'], sgd
'''
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--resize", type=int, default=256)
parser.add_argument("--model", type=str, default='tf_efficientnet_b7')
parser.add_argument("--unfreeze-layer", nargs='+', default=[])
parser.add_argument("--opt", type=str, default='adamw')
parser.add_argument("--opt-param", type=dict, default={})
args = parser.parse_args()


batch_size = 16
epoch_num = 20
eval_step = 1000
l1_weight = 0.001
smooth_weight = 0.1

train_path = r'/home/wuziyang/Projects/data/text_manipulation_detection/dataset'
add_path1 = r'/home/wuziyang/Projects/data/text_manipulation_detection/add_data/train'
add_path2 = r'/home/wuziyang/Projects/data/text_manipulation_detection/add_data/test'

hard_t_path = r'/home/wuziyang/Projects/data/text_manipulation_detection/hard_data/tamper'
hard_ut_path = r'/home/wuziyang/Projects/data/text_manipulation_detection/hard_data/untamper'

# make dataset
tamper_path = os.path.join(train_path, 'tamper')
untamper_path = os.path.join(train_path, 'untamper')

t_trans_size = T.Compose([T.ToTensor(),T.Resize((args.resize, args.resize)), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
un_trans_size = T.Compose([T.ToTensor(),T.Resize((args.resize + 200, args.resize + 200)), T.RandomCrop((args.resize, args.resize)),T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

tamper_list = [os.path.join(tamper_path, p) for p in os.listdir(tamper_path)]
untamper_list = [os.path.join(untamper_path, p) for p in os.listdir(untamper_path)]
add_tamper_list1 = [os.path.join(add_path1, p) for p in os.listdir(add_path1)]
add_tamper_list2 = [os.path.join(add_path2, p) for p in os.listdir(add_path2)]
# add_tamper_hard_list = [os.path.join(hard_t_path, p) for p in os.listdir(hard_t_path)]
add_untamper_hard_list = [os.path.join(hard_ut_path, p) for p in os.listdir(hard_ut_path)]
# add_tamper_list = add_tamper_list1 + add_tamper_list2
add_tamper_list = []
add_untamper_list = add_untamper_hard_list

total_dataset = total_Dataset(tamper_list, untamper_list, add_tamper_inputs=add_tamper_list, 
                                add_untamper_inputs=add_untamper_list, trans=t_trans_size, untrans=un_trans_size, e=smooth_weight)
total_dataloader = DataLoader(dataset=total_dataset, batch_size=batch_size, shuffle=True)

res_all = {}

# prepare model
total_model = timm.create_model(args.model, pretrained=True, num_classes=2)
# 调整finetune层
for name, param in total_model.named_parameters():
    flag = False
    unfreeze_layer = ['blocks.6', 'classifier'] if args.unfreeze_layer == [] else args.unfreeze_layer
    for ufl in unfreeze_layer:
        if ufl in name:
            param.requires_grad = True
            flag = True
            break
    if not flag:
        param.requires_grad = False

total_model = total_model.cuda()
total_model.train()

tamper_criterion = nn.CrossEntropyLoss().cuda()

if args.opt == 'adamw':
    total_optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, total_model.parameters()), lr=1e-5, weight_decay=0.01)
elif args.opt == 'sgd':
    total_optim = torch.optim.SGD(filter(lambda p: p.requires_grad, total_model.parameters()), lr=1e-5, momentum=0.9, weight_decay=2e-5)

# trans = {384:T.Compose([T.Resize(384), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
#          128:T.Compose([T.ToTensor(),T.RandomCrop((128, 128))]),
#          224:T.Compose([T.ToTensor(),T.Resize((224,224))]),
#          299:T.Compose([T.ToTensor(),T.Resize((299,299)),T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
#          256:T.Compose([LoadImage(), T.ToTensor(), T.Resize((256, 256)),T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
#          512:T.Compose([LoadImage(), T.ToTensor(), T.Resize(512, 512),T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
#          1280:T.Compose([T.ToTensor(),T.Resize((128,128))]),
#          0:T.Compose([T.ToTensor(), ])}

# train model
gstep = 0
grecall = 0
gloss = 0
for epoch in range(epoch_num):
    for step, item in enumerate(total_dataloader):
        batch_image, batch_label, batch_imgname = item
        pred = total_model(batch_image)

        # smooth label
        add_label = torch.ones(batch_label.shape[0])
        add_label = (add_label - batch_label).unsqueeze(0)
        batch_label = batch_label.unsqueeze(0)
        batch_label = torch.cat((add_label, batch_label), 0).t().cuda()

        # params = filter(lambda p: p.requires_grad, total_model.parameters())
        # add l1 loss
        # l1_loss = 0.0
        # for p in params:
        #     l1_loss += torch.sum(torch.abs(p))
        loss = tamper_criterion(pred, batch_label)
        # loss = F.smooth_l1_loss(pred, batch_label)
        # loss += l1_weight * l1_loss
        gloss += loss.detach().cpu()

        total_optim.zero_grad()
        loss.backward()
        total_optim.step()

        gstep += 1
        # print(gstep)
        if gstep % eval_step == 0:
            recall = eval(total_model, t_trans_size, smooth_weight, True)
            gloss = gloss / eval_step
            print(f'step {gstep}, epoch {epoch}, recall {recall}, aver loss: {gloss}')
            if recall > grecall:
                grecall = recall
                # if recall > 0.3:
                torch.save(total_model, f'model_tamper_{recall}.pth')
    # hard_example.keys().sort
torch.save(total_model, f'model_tamper_{recall}.pth')
