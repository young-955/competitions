from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import numpy as np
from torchvision import transforms as T
import torch

class untamper_Dataset(Dataset):
    def __init__(
        self,
        tamper_inputs,
        # untamper_inputs,
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
         '256un':T.Compose([T.ToTensor(), T.RandomCrop(384), T.Resize((256, 256)),T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
         1280:T.Compose([T.ToTensor(),T.Resize((128,128))]),
         0:T.Compose([T.ToTensor(), ])}

        self.tamper_inputs = tamper_inputs
        # self.untamper_inputs = untamper_inputs
        # self.inputs = self.untamper_inputs + self.tamper_inputs
        self.inputs = self.untamper_inputs
        # self.labels = np.append(np.zeros(len(self.untamper_inputs)), np.ones(len(self.tamper_inputs)))
        self.labels = np.zeros(len(self.untamper_inputs))

    def __len__(self):
        # return len(self.tamper_inputs) + len(self.untamper_inputs)
        return len(self.untamper_inputs)

    def __getitem__(self, index):
        if self.labels[index] == 0:
            img = self.trans['256un'](Image.open(self.inputs[index]))
        else:
            img = self.trans[256](Image.open(self.inputs[index]))
        img = torch.Tensor(img).cuda()
        return img, self.labels[index], self.inputs[index]

class tamper_Dataset(Dataset):
    def __init__(
        self,
        tamper_inputs,
    ):
        super().__init__()

        self.trans = {384:T.Compose([T.ToTensor(), T.Resize(384), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
         128:T.Compose([T.ToTensor(),T.RandomCrop((128, 128))]),
         224:T.Compose([T.ToTensor(),T.Resize((224,224))]),
         299:T.Compose([T.ToTensor(),T.Resize((299,299)),T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
         256:T.Compose([T.ToTensor(), T.Resize((256, 256)),T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
         1280:T.Compose([T.ToTensor(),T.Resize((128,128))]),
         0:T.Compose([T.ToTensor(), ])}

        self.tamper_inputs = tamper_inputs
        self.inputs = self.tamper_inputs
        self.labels = np.ones(len(self.tamper_inputs))

    def __len__(self):
        return len(self.tamper_inputs)

    def __getitem__(self, index):
        img = self.trans[256](Image.open(self.inputs[index]))
        img = torch.Tensor(img).cuda()
        return img, self.labels[index], self.inputs[index]

class total_Dataset(Dataset):
    '''
        e: label smooth
    '''
    def __init__(
        self,
        tamper_inputs,
        untamper_inputs,
        add_tamper_inputs=[],
        add_untamper_inputs=[],
        trans=None,
        untrans=None,
        e=0
    ):
        super().__init__()

        self.trans = {384:T.Compose([T.ToTensor(), T.Resize(384), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
         128:T.Compose([T.ToTensor(),T.RandomCrop((128, 128))]),
         224:T.Compose([T.ToTensor(),T.Resize((224,224))]),
         299:T.Compose([T.ToTensor(),T.Resize((299,299)),T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
         256:T.Compose([T.ToTensor(), T.Resize((256, 256)),T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
         '256un':T.Compose([T.ToTensor(), T.RandomCrop(384), T.Resize((256, 256)),T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
         1280:T.Compose([T.ToTensor(),T.Resize((128,128))]),
         0:T.Compose([T.ToTensor(), ])}

        if trans:
            self.t_trans = trans
        else:
            self.t_trans = self.trans[256]
        if untrans:
            self.t_untrans = untrans
        else:
            self.t_untrans = self.trans['256un']

        self.tamper_inputs = tamper_inputs + add_tamper_inputs
        self.untamper_inputs = untamper_inputs + add_untamper_inputs
        self.inputs = self.untamper_inputs + self.tamper_inputs
        if e == 0:
            self.labels = np.append(np.zeros(len(self.untamper_inputs)), np.ones(len(self.tamper_inputs)))
        else:
            self.labels = np.append(np.full(len(self.untamper_inputs), e), np.full(len(self.tamper_inputs), 1-e))

    def __len__(self):
        return len(self.tamper_inputs) + len(self.untamper_inputs)

    def __getitem__(self, index):
        if self.labels[index] == 0:
            img = self.t_untrans(Image.open(self.inputs[index]))
        else:
            img = self.t_trans(Image.open(self.inputs[index]))
        img = torch.Tensor(img).cuda()
        return img, self.labels[index], self.inputs[index]
