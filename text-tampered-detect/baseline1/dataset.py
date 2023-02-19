from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import numpy as np
from torchvision import transforms as T
import torch

class tamper_Dataset(Dataset):

    def __init__(
        self,
        temper_inputs,
        untemper_inputs,
        # temper_transform,
        # untemper_transform,
        # labels,
    ):
        super().__init__()

        self.trans = {384:T.Compose([T.ToTensor(), T.Resize(384), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
         128:T.Compose([T.ToTensor(),T.RandomCrop((128, 128))]),
         224:T.Compose([T.ToTensor(),T.Resize((224,224))]),
         299:T.Compose([T.ToTensor(),T.Resize((299,299)),T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
         256:T.Compose([T.ToTensor(), T.Resize((256, 256)),T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
         1280:T.Compose([T.ToTensor(),T.Resize((128,128))]),
         0:T.Compose([T.ToTensor(), ])}


        self.tamper_inputs = temper_inputs
        self.untamper_inputs = untemper_inputs
        self.inputs = self.tamper_inputs + self.untamper_inputs
        # self.temper_transform = temper_transform
        # self.untemper_transform = untemper_transform
        self.labels = np.append(np.ones(len(self.tamper_inputs)), np.zeros(len(self.untamper_inputs)))

    def __len__(self):
        return len(self.tamper_inputs) + len(self.untamper_inputs)

    def __getitem__(self, index):
        img = self.trans[384](Image.open(self.inputs[index]))
        img = torch.Tensor(img).cuda()
        return img, self.labels[index]
