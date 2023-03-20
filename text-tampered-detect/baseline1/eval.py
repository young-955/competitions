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
import shutil
from sklearn.cluster import KMeans

eval_tamper_path = '/home/wuziyang/Projects/data/text_manipulation_detection/dataset/test/0/data.npy'
eval_untamper_path = '/home/wuziyang/Projects/data/text_manipulation_detection/dataset/test/1/data.npy'

t_data = np.load(eval_tamper_path)
ut_data = np.load(eval_untamper_path)

def score_cls(submission_path, labels):
    """
    submission_path: 'competition1/815903/815903_2023-02-13 18:48:29.csv'
    labels: labels = pd.read_csv('competition1/labels.txt',
                         delim_whitespace=True, header=None).to_numpy()
    赛道1评分
    """
    submission = pd.read_csv(submission_path, delim_whitespace=True, header=None).to_numpy()
    tampers = labels[labels[:, 1] == 1]
    untampers = labels[labels[:, 1] == 0]
    pred_tampers = submission[np.in1d(submission[:, 0], tampers[:, 0])]
    pred_untampers = submission[np.in1d(submission[:, 0], untampers[:, 0])]

    thres = np.percentile(pred_untampers[:, 1], np.arange(90, 100, 1))
    recall = np.mean(np.greater(pred_tampers[:, 1][:, np.newaxis], thres).mean(axis=0))
    return recall * 100

hard_examples = {}

def eval(model, trans=tamper_transform, e=0, hard=False):
    print('start evaluation')
    model.eval()
    # loss = 0.0
    hard_limit = 10
    hard_t_example = {}
    hard_ut_example = {}
    hard_min_value = 0.0
    hard_max_value = 1.0

    # tamper label 1
    tamper_pred = []
    for i in t_data:
        img = trans(i)
        img = img.unsqueeze(0).float().cuda().to(torch.float32)
        res = model(img)
        prob = torch.nn.functional.softmax(res)
        # loss += abs(prob[:,1].detach().cpu().numpy()[0] - 1) ** 2
        pred_res = prob[:,1].detach().cpu().numpy()[0]
        tamper_pred.append(pred_res)

        if hard:
            if pred_res < hard_max_value:
                if len(hard_t_example) < hard_limit:
                    hard_t_example[pred_res] = i
                    hard_max_value = max(hard_t_example.keys())
                else:
                    del hard_t_example[hard_max_value]
                    hard_t_example[pred_res] = i
                    hard_max_value = max(hard_t_example.keys())

    # untamper label 0
    untamper_pred = []
    for i in ut_data:
        img = trans(i)
        img = img.unsqueeze(0).float().cuda().to(torch.float32)
        res = model(img)
        prob = torch.nn.functional.softmax(res)
        # loss += abs(prob[:,1].detach().cpu().numpy()[0]) ** 2
        pred_res = prob[:,1].detach().cpu().numpy()[0]
        if pred_res > 1-e or pred_res < e:
            pred_res = 0.5
        untamper_pred.append(pred_res)

        if hard:
            if pred_res > hard_min_value:
                if len(hard_ut_example) < hard_limit:
                    hard_ut_example[pred_res] = i
                    hard_min_value = min(hard_ut_example.keys())
                else:
                    del hard_ut_example[hard_min_value]
                    hard_ut_example[pred_res] = i
                    hard_min_value = min(hard_ut_example.keys())

    thres = np.percentile(np.array(untamper_pred), np.arange(90, 100, 1))
    recall = np.mean(np.greater(np.array(tamper_pred)[:, np.newaxis], thres).mean(axis=0))

    if hard:
        for v in hard_ut_example.values():
            if v in hard_examples:
                hard_examples[v] += 1
            else:
                hard_examples[v] = 1
        for v in hard_t_example.values():
            if v in hard_examples:
                hard_examples[v] += 1
            else:
                hard_examples[v] = 1

        print(f'hard un tamper imgs: {hard_ut_example}')
        print(f'hard tamper imgs: {hard_t_example}')
        print(f'hard examples: {hard_examples}')
    model.train()
    return recall


def eval_train(model, trans=tamper_transform, e=0):
    model.eval()
    # loss = 0.0

    # tamper label 1
    tamper_pred = []
    for i in t_data:
        img = trans(i)
        img = img.unsqueeze(0).float().cuda().to(torch.float32)
        res = model(img)
        prob = torch.nn.functional.softmax(res)
        # loss += abs(prob[:,1].detach().cpu().numpy()[0] - 1) ** 2
        tamper_pred.append(prob[:,1].detach().cpu().numpy()[0])

    # untamper label 0
    hard_limit = 10
    hard_example = {}
    hard_min_value = 0.0
    untamper_pred = []
    for i in ut_data:
        img = trans(i)
        img = img.unsqueeze(0).float().cuda().to(torch.float32)
        res = model(img)
        prob = torch.nn.functional.softmax(res)
        # loss += abs(prob[:,1].detach().cpu().numpy()[0]) ** 2
        pred_res = prob[:,1].detach().cpu().numpy()[0]
        if pred_res > 1-e or pred_res < e:
            pred_res = 0.5
        untamper_pred.append(pred_res)
        if pred_res > hard_min_value:
            if len(hard_example) < hard_limit:
                hard_example[pred_res] = i
                hard_min_value = min(hard_example.keys())
            else:
                del hard_example[hard_min_value]
                hard_example[pred_res] = i
                hard_min_value = min(hard_example.keys())

    thres = np.percentile(np.array(untamper_pred), np.arange(90, 100, 1))
    recall = np.mean(np.greater(np.array(tamper_pred)[:, np.newaxis], thres).mean(axis=0))

    print(f'hard imgs: {hard_example}')
    model.train()
    return recall

def eval_pseudo(model, trans=tamper_transform, hard=False, pseudo_path=""):
    print('start evaluation')
    model.eval()
    # loss = 0.0
    hard_limit = 10
    hard_t_example = {}
    hard_ut_example = {}
    hard_min_value = 0.0
    hard_max_value = 1.0

    pseudo_up_limit = 0.92
    pseudo_down_limit = 0.02

    predict_path = '/home/wuziyang/Projects/data/text_manipulation_detection/test/imgs'
    test_data = np.array([os.path.join(predict_path, i) for i in os.listdir(predict_path)])
    pset_data = np.array([os.path.join(os.path.join(pseudo_path, 'tamper'), i) for i in os.listdir(os.path.join(pseudo_path, 'tamper'))])
    pseut_data = np.array([os.path.join(os.path.join(pseudo_path, 'untamper'), i) for i in os.listdir(os.path.join(pseudo_path, 'untamper'))])

    tamper_pred = []
    untamper_pred = []
    unlabel_pred = []

    for i in test_data:
        img = trans(i)
        img = img.unsqueeze(0).float().cuda().to(torch.float32)
        res = model(img)
        prob = torch.nn.functional.softmax(res)
        pred_res = prob[:,1].detach().cpu().numpy()[0]
        unlabel_pred.append(pred_res)

        if pred_res < pseudo_down_limit:
            name = i.split('/')[-1]
            shutil.copyfile(i, os.path.join(pseudo_path, f'untamper/{name}'))
        if pred_res > pseudo_up_limit:
            name = i.split('/')[-1]
            shutil.copyfile(i, os.path.join(pseudo_path, f'tamper/{name}'))

        if len(unlabel_pred) > 2:
            cluster_res = KMeans(n_clusters=2,random_state=0).fit(np.array(unlabel_pred).reshape(-1, 1))
            cluster_dis = abs(cluster_res.cluster_centers_[0][0] - cluster_res.cluster_centers_[1][0])
        else:
            cluster_dis = 0

    for i in pset_data:
        img = trans(i)
        img = img.unsqueeze(0).float().cuda().to(torch.float32)
        res = model(img)
        prob = torch.nn.functional.softmax(res)
        pred_res = prob[:,1].detach().cpu().numpy()[0]
        tamper_pred.append(pred_res)

        if hard:
            if pred_res < hard_max_value:
                if len(hard_t_example) < hard_limit:
                    hard_t_example[pred_res] = i
                    hard_max_value = max(hard_t_example.keys())
                else:
                    del hard_t_example[hard_max_value]
                    hard_t_example[pred_res] = i
                    hard_max_value = max(hard_t_example.keys())
    for i in pseut_data:
        img = trans(i)
        img = img.unsqueeze(0).float().cuda().to(torch.float32)
        res = model(img)
        prob = torch.nn.functional.softmax(res)
        pred_res = prob[:,1].detach().cpu().numpy()[0]
        untamper_pred.append(pred_res)

        if hard:
            if pred_res > hard_min_value:
                if len(hard_ut_example) < hard_limit:
                    hard_ut_example[pred_res] = i
                    hard_min_value = min(hard_ut_example.keys())
                else:
                    del hard_ut_example[hard_min_value]
                    hard_ut_example[pred_res] = i
                    hard_min_value = min(hard_ut_example.keys())

    thres = np.percentile(np.array(untamper_pred), np.arange(90, 100, 1))
    recall = np.mean(np.greater(np.array(tamper_pred)[:, np.newaxis], thres).mean(axis=0))

    if hard:
        for v in hard_ut_example.values():
            if v in hard_examples:
                hard_examples[v] += 1
            else:
                hard_examples[v] = 1
        for v in hard_t_example.values():
            if v in hard_examples:
                hard_examples[v] += 1
            else:
                hard_examples[v] = 1

        print(f'hard un tamper imgs: {hard_ut_example}')
        print(f'hard tamper imgs: {hard_t_example}')
        print(f'hard examples: {hard_examples}')
    model.train()
    return recall, cluster_dis

def eval_hard(model_path, path):
    trans = T.Compose([LoadImage(), T.ToTensor(),T.Resize((512, 512)), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    model = torch.load(model_path).cuda()
    model.eval()
    with open(path, 'r') as f:
        data = f.readlines()
    f.close()

    with open('t', 'w') as f:
        for d in data:
            path = d.replace('\'', '').split(':')[0]
            img = trans(path).unsqueeze(0).float().cuda().to(torch.float32)
            res = model(img)
            prob = torch.nn.functional.softmax(res)
            pred_res = prob[:,1].detach().cpu().numpy()[0]
            f.write(f'{path}: {pred_res}')
            f.write('\n')
    f.close()

def gen_test_data():
    # 数据增强
    from PIL import Image
    dataset_path = r'/home/wuziyang/Projects/data/text_manipulation_detection/dataset'
    tamper_dataset_path = os.path.join(dataset_path, 'tamper')
    untamper_dataset_path = os.path.join(dataset_path, 'untamper')

    for i in t_data:
        img = Image.open(i)
        img.save(os.path.join(tamper_dataset_path, i))
        i1 = data_aug1(i)
        i1.save(os.path.join(tamper_dataset_path, i.split('/')[-1].split('.')[0] + '1.png'))
        i2 = data_aug2(i)
        i2.save(os.path.join(tamper_dataset_path, i.split('/')[-1].split('.')[0] + '2.png'))
        i3 = data_aug3(i)
        i3.save(os.path.join(tamper_dataset_path, i.split('/')[-1].split('.')[0] + '3.png'))

    print('train tamper data ready')

    for i in ut_data:
        img = Image.open(i)
        img.save(os.path.join(untamper_dataset_path, i))
        i1 = data_aug1(i)
        i1.save(os.path.join(untamper_dataset_path, i.split('/')[-1].split('.')[0] + '1.png'))
        i2 = data_aug2(i)
        i2.save(os.path.join(untamper_dataset_path, i.split('/')[-1].split('.')[0] + '2.png'))
        i3 = data_aug3(i)
        i3.save(os.path.join(untamper_dataset_path, i.split('/')[-1].split('.')[0] + '3.png'))

    print('train untamper data ready')


if __name__ == "__main__":
    gen_test_data()