import os
from torchvision import transforms
from PIL import Image

class LoadImage:
    def __call__(self, results):
        img = Image.open(results)
        return img


gen_num = 10

save_un_path = r'/home/wuziyang/Projects/data/text_manipulation_detection/hard_data/untamper'
save_t_path = r'/home/wuziyang/Projects/data/text_manipulation_detection/hard_data/tamper'

def aug_hard_un_data():
    trans = transforms.Compose([
        LoadImage(), 
        transforms.Resize((1600, 1600)),
        transforms.RandomCrop((1024, 1024)),
        transforms.RandomRotation(180)
    ])

    with open('hard_un_list', 'r') as f:
        data = f.readlines()
    f.close()

    for d in data:
        for i in range(20):
            res = trans(d.replace('\n', '').replace('\'', ''))
            res.save(os.path.join(save_un_path, d.split('/')[-1].split('.')[0] + f'_hard{i}.png'))

if __name__ == '__main__':
    aug_hard_un_data()