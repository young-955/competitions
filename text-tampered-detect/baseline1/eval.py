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

def score_cls(submission_path, labels):
    """
    submission_path: 'competition1/815903/815903_2023-02-13 18:48:29.csv'
    labels: labels = pd.read_csv('competition1/labels.txt',
                         delim_whitespace=True, header=None).to_numpy()
    赛道1评分
    """
    submission = pd.read_csv(
        submission_path, delim_whitespace=True, header=None).to_numpy()
    tampers = labels[labels[:, 1] == 1]
    untampers = labels[labels[:, 1] == 0]
    pred_tampers = submission[np.in1d(submission[:, 0], tampers[:, 0])]
    pred_untampers = submission[np.in1d(submission[:, 0], untampers[:, 0])]

    thres = np.percentile(pred_untampers[:, 1], np.arange(90, 100, 1))
    recall = np.mean(np.greater(pred_tampers[:, 1][:, np.newaxis], thres).mean(axis=0))
    return recall * 100

def eval(model):
    t_data = np.load(eval_tamper_path)
    ut_data = np.load(eval_untamper_path)

    # loss = 0.0

    # tamper label 1
    tamper_pred = []
    for i in t_data:
        img = tamper_transform(i)
        img = img.unsqueeze(0).float().cuda().to(torch.float32)
        res = model(img)
        prob = torch.nn.functional.softmax(res)
        # loss += abs(prob[:,1].detach().cpu().numpy()[0] - 1) ** 2
        tamper_pred.append(prob[:,1].detach().cpu().numpy()[0])

    # untamper label 0
    untamper_pred = []
    for i in ut_data:
        print(i)
        img = tamper_transform(i)
        img = img.unsqueeze(0).float().cuda().to(torch.float32)
        res = model(img)
        prob = torch.nn.functional.softmax(res)
        # loss += abs(prob[:,1].detach().cpu().numpy()[0]) ** 2
        untamper_pred.append(prob[:,1].detach().cpu().numpy()[0])

    thres = np.percentile(np.array(untamper_pred), np.arange(90, 100, 1))
    recall = np.mean(np.greater(np.array(tamper_pred)[:, np.newaxis], thres).mean(axis=0))
    # print(f'recall:{recall}')
    return recall

    # print(f'loss: {loss}')

if __name__ == "__main__":
    model = torch.load('pretrain_baseline.pth',map_location=torch.device('cpu'))
    model.cuda()
    model.eval()
    eval(model)
    # t_data = np.load(eval_tamper_path)
    # print(t_data.shape)