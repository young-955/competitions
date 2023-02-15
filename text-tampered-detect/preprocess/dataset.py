from torch.utils.data import Dataset

class temper_Dataset(Dataset):

    def __init__(
        self,
        temper_inputs,
        untemper_inputs,
        temper_transform,
        untemper_transform,
        labels,
    ):
        super().__init__()

        self.temper_inputs = temper_inputs
        self.untemper_inputs = untemper_inputs
        self.temper_transform = temper_transform
        self.untemper_transform = untemper_transform
        self.labels = labels

    def __len__(self):
        return len(self.temper_inputs) + len(self.untemper_inputs)

    def __getitem__(self, index):
        
        example = self.examples[index]
        label = None
        if example.label is not None and example.label != "":
            label = self.label2id[example.label]

        # tokenize
        text_a_token_ids, text_a_attention_mask = self._tokenize(example.text_a)
        text_b_token_ids, text_b_attention_mask = self._tokenize(example.text_b)

        return {'text_a_input_ids': text_a_token_ids, 'text_b_input_ids': text_b_token_ids, 
                'text_a_attention_mask': text_a_attention_mask, 'text_b_attention_mask': text_b_attention_mask, 'label': label}

