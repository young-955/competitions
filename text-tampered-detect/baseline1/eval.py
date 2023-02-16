import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
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
sys.path.append('../')
from preprocess.transform_tamplate import *

eval_tamper_path = '/home/wuziyang/Projects/data/text_manipulation_detection/dataset/test/0/data.npy'
eval_untamper_path = '/home/wuziyang/Projects/data/text_manipulation_detection/dataset/test/1/data.npy'


def eval(model):
    t_data = np.load(eval_tamper_path)
    ut_data = np.load(eval_untamper_path)

    loss = 0.0

    # tamper label 1
    for i in t_data:
        print(i)
        img = temper_transform(i)
        img = img.unsqueeze(0).float().cuda().to(torch.float32)
        res = model(img)
        prob = torch.nn.functional.softmax(res)
        loss += abs(prob[:,1].detach().cpu().numpy()[0] - 1) ** 2

    # untamper label 0
    for i in ut_data:
        print(i)
        img = temper_transform(i)
        img = img.unsqueeze(0).float().cuda().to(torch.float32)
        res = model(img)
        prob = torch.nn.functional.softmax(res)
        loss += abs(prob[:,1].detach().cpu().numpy()[0]) ** 2

    print(f'loss: {loss}')

if __name__ == "__main__":
    model = torch.load('pretrain_baseline.pth',map_location=torch.device('cpu'))
    model.cuda()
    model.eval()
    eval(model)
    # t_data = np.load(eval_tamper_path)
    # print(t_data.shape)