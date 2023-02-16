import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from models.mvssnet import get_mvss

save_root_path='./res'

# 加载模型
model = get_mvss(sobel=True, n_input=3, constrain=True)
model_dict = torch.load('./mvss_pretrain.pth',map_location=torch.device('cpu'))
model.load_state_dict(model_dict['state_dict'])

model.eval()
model.to('cpu')

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 加载图像并进行预测
img_paths='./0002.jpg'
image = cv2.imread(img_paths)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
l,w,_=image.shape
image_tensor = transform(Image.fromarray(image)).unsqueeze(0)
with torch.no_grad():
    _,outputs = model(image_tensor)
    outputs = torch.sigmoid(outputs)
outputs = outputs.data.cpu().numpy()[:, 0, :, :]
outputs = np.array(outputs * 255, dtype=int)
img_name = img_paths.split('/')[-1]  # win上面需要把 / 改为 \\
img_name = img_name.replace('jpg','png')
save_path = os.path.join(save_root_path, img_name)
mask = outputs[0]
mask = mask.astype(np.uint8)
mask = cv2.resize(mask,(w,l))

# 保存概率图
cv2.imwrite(save_path, mask)
