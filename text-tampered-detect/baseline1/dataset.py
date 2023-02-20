from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import numpy as np

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

        self.tamper_inputs = temper_inputs
        self.untamper_inputs = untemper_inputs
        self.inputs = self.tamper_inputs + self.untamper_inputs
        # self.temper_transform = temper_transform
        # self.untemper_transform = untemper_transform
        self.labels = np.append(np.ones(len(self.tamper_inputs)), np.zeros(len(self.untamper_inputs)))

    def __len__(self):
        return len(self.tamper_inputs) + len(self.untamper_inputs)

    def __getitem__(self, index):
        return np.array(Image.open(self.inputs[index])), self.labels[index]
