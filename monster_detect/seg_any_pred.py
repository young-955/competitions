from seg.segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
from PIL import Image
import copy
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

# load model
sam = sam_model_registry["default"](checkpoint="./seg/model/sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)
mask_generator.predictor.model.cuda()

# 物料是否偏移
class1_train_data = read_text(r'./data/data/class1/class1_train.txt')
class2_train_data = read_text(r'./data/data/class2/class2_train.txt')
# 全部物料偏移
class3_train_data = read_csv(r'./data/data/class3/train_infos.csv')
# 全部物料不偏移
normal_train_data = read_text(r'./data/data/normal/normal_train.txt')


def save_seg(image, masks):
    cv2.imwrite(f'./test/ori.png', image)
    max_area = [0, 0]
    next_area = [0, 0]
    for i, m in enumerate(masks):
    #     if m['area'] > next_area[1]:
    #         if m['area'] > max_area[1]:
    #             next_area = copy.deepcopy(max_area)
    #             max_area[0] = i
    #             max_area[1] = m['area']
    #         else:
    #             next_area[0] = i
    #             next_area[1] = m['area']
    # print(max_area)
    # print(next_area)
        cv2.imwrite(f'./test/{i}.png', image[m['bbox'][1]: m['bbox'][1] + m['bbox'][3], m['bbox'][0]:m['bbox'][0] + m['bbox'][2], :])

# 分割
def seg_img(model, image):
    return model.generate(image)

# 获取最大和次大的mask,分别表示传送带和物料带
def get_max_mask(masks):
    max_area = [0, 0]
    next_area = [0, 0]
    for i, m in enumerate(masks):
        area = m['bbox'][2] * m['bbox'][3]
        if area > next_area[1]:
            if area > max_area[1]:
                next_area = copy.deepcopy(max_area)
                max_area[0] = i
                max_area[1] = area
            else:
                next_area[0] = i
                next_area[1] = area
    return masks[max_area[0]], masks[next_area[0]]
    
# 计算两端偏移像素差值和比例
def get_offset_diff(max_mask, next_mask):
    offset1 = next_mask['bbox'][0] - max_mask['bbox'][0]
    offset2 = max_mask['bbox'][0] + max_mask['bbox'][2] - next_mask['bbox'][0] - next_mask['bbox'][2]

    abs_off_diff = abs(offset1 - offset2)
    off_diff_rate = abs_off_diff / (offset1 + offset2) * 2

    return abs_off_diff, off_diff_rate

# 判断是否偏移
def judge_offset(masks):
    max_mask, next_mask = get_max_mask(masks)    
    return get_offset_diff(max_mask, next_mask)

# 跑偏检测测试
def offset_train_test(model, data):
    print('start process ...')
    print(f'total data: {len(data)}')
    type0_diff = []
    type3_diff = []
    try:
        for i, d in enumerate(data):
            print(f'now {i+1}th data process, file name: {d[0]}')
            image = cv2.imread(d[0])
            masks = seg_img(model, image)

            abs_off_diff, off_diff_rate = judge_offset(masks)
            if d[1] == 0:
                type0_diff.append([d[0], abs_off_diff, off_diff_rate])
            elif d[1] == 3:
                type3_diff.append([d[0], abs_off_diff, off_diff_rate])
            else:
                print(f'error:{d}')
    except:
        pd.DataFrame(type0_diff).to_csv(r'./type0diff.csv')
        pd.DataFrame(type3_diff).to_csv(r'./type3diff.csv')

    pd.DataFrame(type0_diff).to_csv(r'./type0diff.csv')
    pd.DataFrame(type3_diff).to_csv(r'./type3diff.csv')


# 异物检测及跑偏检测
def pipeline(model, data):

    return

def main():
    class1_train_data
    class2_train_data
    class3_train_data
    normal_train_data
    input_data = []
    input_data.extend(class1_train_data)
    input_data.extend(class2_train_data)
    input_data.extend(class3_train_data)
    input_data.extend(normal_train_data)
    offset_train_test(mask_generator, input_data)

if __name__ == '__main__':
    main()