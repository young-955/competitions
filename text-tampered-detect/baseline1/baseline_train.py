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

res_all = {}

import timm
model = timm.create_model('tf_efficientnet_b1', pretrained=True, num_classes=2)
# model = torch.load('pretrain_baseline.pth',map_location=torch.device('cpu'))
model = model.cuda()
# model.eval()

trans = {384:T.Compose([T.ToTensor(),T.Resize(384),T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
         128:T.Compose([T.ToTensor(),T.RandomCrop((128, 128))]),
         224:T.Compose([T.ToTensor(),T.Resize((224,224))]),
         299:T.Compose([T.ToTensor(),T.Resize((299,299)),T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
         256:T.Compose([T.ToTensor(), T.Resize((256, 256)),T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
         1280:T.Compose([T.ToTensor(),T.Resize((128,128))]),
         0:T.Compose([T.ToTensor(), ])}

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

