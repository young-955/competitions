import pandas as pd
import os
import numpy as np

def read_text(path):
    res = []
    base_path = os.path.dirname(os.path.dirname(path))
    with open(path, 'r') as f:
        for p in f.readlines():
            p = p.replace('\n', '').split(',')
            p[0] = os.path.join(base_path, p[0])
            p[1] = int(p[1])
            res.append(p)
    return res

def read_csv(path):
    res = []
    base_path = os.path.join(os.path.dirname(path), 'train')
    csv_p = pd.read_csv(path)[['filename', 'label']].drop_duplicates().values
    for p in csv_p:
        p[0] = os.path.join(base_path, p[0])
        res.append(p)
    return res

# 物料是否偏移
class1_train_data = read_text(r'./data/data/class1/class1_train.txt')
class2_train_data = read_text(r'./data/data/class2/class2_train.txt')
# 全部物料偏移
class3_train_data = read_csv(r'./data/data/class3/train_infos.csv')
# 全部物料不偏移
normal_train_data = read_text(r'./data/data/normal/normal_train.txt')
