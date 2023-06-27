from seg.segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
from config import *
import numpy as np
import cv2
from utils import *

sam = sam_model_registry["default"](checkpoint="./seg/model/sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(sam)
mask_generator.predictor.model.cuda()

def pred_test(model, data):
    input_point = np.array([[1280, 720]])
    input_label = np.array([1])
    
    for i, d in enumerate(data):
        image = cv2.imread(d[0])

        model.set_image(image)

        masks, scores, logits  = model.predict(point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )


    return model

def mask_test(model, data):
    type0_diff = []
    type3_diff = []
    try:
        for i, d in enumerate(data):
            print(d[0])
            image = cv2.imread(d[0])
            print(image.shape)

            masks = model.generate(image)

            print('finish generate mask')
            res = get_anns(masks)
            
            if res[2]['bbox'][0] < res[3]['bbox'][0]:
                off1 = res[0]['bbox'][0] - res[2]['bbox'][0]
                off2 = res[3]['bbox'][0] + res[3]['bbox'][2] - res[0]['bbox'][0] + res[0]['bbox'][2]
            else:
                off1 = res[0]['bbox'][0] - res[3]['bbox'][0]
                off2 = res[2]['bbox'][0] + res[2]['bbox'][2] - res[0]['bbox'][0] + res[0]['bbox'][2]

            offset = abs(off2 - off1)
            offset_rate = offset / ((off1 + off2) / 2)
        
            if d[1] == 0:
                type0_diff.append([d[0], offset, offset_rate])
            elif d[1] == 3:
                type3_diff.append([d[0], offset, offset_rate])
            else:
                print(f'error:{d}')
    except:
        pd.DataFrame(type0_diff).to_csv(r'./type0diff.csv')
        pd.DataFrame(type3_diff).to_csv(r'./type3diff.csv')

    pd.DataFrame(type0_diff).to_csv(r'./type0diff.csv')
    pd.DataFrame(type3_diff).to_csv(r'./type3diff.csv')


def main():
    input_data = []
    input_data.extend(class1_train_data)
    input_data.extend(class2_train_data)
    input_data.extend(class3_train_data)
    input_data.extend(normal_train_data)
    mask_test(mask_generator, input_data)



if __name__ == '__main__':
    main()