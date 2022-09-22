# %%
import os
import json
import math
import numpy as np
import dataclasses
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import List, Union, Dict, Any, Mapping, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import set_seed
from transformers.data.processors.utils import InputExample, InputFeatures
from transformers.data import DefaultDataCollator
from transformers import PreTrainedTokenizer, AutoTokenizer, PreTrainedModel, BertForSequenceClassification
from transformers.trainer_pt_utils import get_parameter_names
from transformers.optimization import get_linear_schedule_with_warmup, AdamW

set_seed(2022)

@dataclass
class DataTrainingArguments:

    model_dir: str = field(
        default='chinese-bert-wwm-ext',
        metadata={'help': 'The pretrained model directory'}
    )
    data_dir: str = field(
        default='./data',
        metadata={'help': 'The data directory'}
    )
    max_length: int = field(
        default=64,
        metadata={'help': 'Maximum sequence length allowed to input'}
    )

    def __str__(self):
        self_as_dict = dataclasses.asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"
        
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass
class TrainingArguments:

    output_dir: str = field(
        default='./model',
        metadata={'help': 'The output directory where the model predictions and checkpoints will be written.'}
    )
    train_batch_size: int = field(
        default=16,
        metadata={'help': 'batch size for training'}
    )
    eval_batch_size: int = field(
        default=32,
        metadata={'help': 'batch size for evaluation'}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={'help': 'Number of updates steps to accumulate before performing a backward/update pass.'}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "The total number of training epochs"}
    )
    learning_rate: float = field(
        default=3e-5,
        metadata={'help': '"The initial learning rate for AdamW.'}
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay for AdamW"}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    dataloader_num_workers: int = field(
        default=4,
        metadata={"help": "Number of subprocesses to use for data loading (PyTorch only)"}
    )
    
    logging_steps: int = field(
        default=100,
        metadata={'help': 'logging states every X updates steps.'}
    )
    eval_steps: int = field(
        default=250,
        metadata={'help': 'Run an evaluation every X steps.'}
    )
    device: str = field(
        default='cpu',
        metadata={"help": 'The device used for training'}
    )

    def get_warmup_steps(self, num_training_steps):
        return int(num_training_steps * self.warmup_ratio)

    def __str__(self):
        self_as_dict = dataclasses.asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"
        
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"

class QQRProcessor:
    TASK = 'KUAKE-QQR'

    def __init__(self, data_dir):
        self.task_dir = os.path.join(data_dir)

    def get_train_examples(self):
        return self._create_examples(os.path.join(self.task_dir, f'{self.TASK}_train.json'))

    def get_dev_examples(self):
        return self._create_examples(os.path.join(self.task_dir, f'{self.TASK}_dev.json'))

    def get_test_examples(self):
        return self._create_examples(os.path.join(self.task_dir, f'{self.TASK}_test.json'))

    def get_labels(self):
        return ["0", "1", "2"]

    def _create_examples(self, data_path):

        # 读入文件
        with open(data_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)

        examples = []
        for sample in samples:
            guid = sample['id']
            text_a = sample['query1']
            text_b = sample['query2']
            label = sample.get('label', None)

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


class ClassificationDataset(Dataset):

    def __init__(
        self,
        examples: List[InputExample],
        label_list: List[Union[str, int]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
        processor = None
    ):
        super().__init__()

        self.examples = examples
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.processor = processor

        self.label2id = {label: idx for idx, label in enumerate(label_list)}
        self.id2label = {idx: label for idx, label in enumerate(label_list)}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index) -> InputFeatures:
        
        example = self.examples[index]
        label = None
        if example.label is not None and example.label != "":
            label = self.label2id[example.label]

        inputs = self.tokenizer(
            text=example.text_a,
            text_pair=example.text_b,
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )

        feature = InputFeatures(**inputs, label=label)

        return feature

def create_optimizer_and_scheduler(
    args: TrainingArguments,
    model: PreTrainedModel,
    num_training_steps: int,
):
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_training_steps=num_training_steps, 
        num_warmup_steps=args.get_warmup_steps(num_training_steps)
    )

    return optimizer, scheduler

def _prepare_input(data: Union[torch.Tensor, Any], device: str = 'cuda'):
    # 将准备输入模型中的数据转到GPU上
    if isinstance(data, Mapping):
        return type(data)({k: _prepare_input(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(_prepare_input(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = dict(device=device)
        return data.to(**kwargs)
    return data


def simple_accuracy(preds, labels):

    return (preds == labels).mean()

def evaluate(
    args: TrainingArguments,
    model: PreTrainedModel,
    eval_dataloader
):
    model.eval()
    loss_list = []
    preds_list = []
    labels_list = []

    for item in eval_dataloader:
        inputs = _prepare_input(item, device=args.device)

        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
            loss = outputs.loss
            loss_list.append(loss.detach().cpu().item())

            preds = torch.argmax(outputs.logits.cpu(), dim=-1).numpy()
            preds_list.append(preds)

            labels_list.append(inputs['labels'].cpu().numpy())
    
    preds = np.concatenate(preds_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    loss = np.mean(loss_list)
    accuracy = simple_accuracy(preds, labels)

    model.train()

    return loss, accuracy

def train(
    args: TrainingArguments,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset,
    dev_dataset,
    data_collator,
):

    # initialize dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=args.train_batch_size,
        shuffle=True,
        # num_workers=args.dataloader_num_workers,
        collate_fn=data_collator
    )
    dev_dataloader = DataLoader(
        dataset=dev_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        # num_workers=args.dataloader_num_workers,
        collate_fn=data_collator
    )

    num_examples = len(train_dataloader.dataset)
    total_train_batch_size = args.gradient_accumulation_steps * args.train_batch_size
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    
    max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
    num_train_epochs = math.ceil(args.num_train_epochs)
    num_train_samples = len(train_dataset) * args.num_train_epochs

    optimizer, lr_scheduler = create_optimizer_and_scheduler(
        args, model, num_training_steps=max_steps
    )

    print("***** Running training *****")
    print(f"  Num examples = {num_examples}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {max_steps}")

    model.zero_grad()
    model.train()
    t_loss = 0.0
    global_steps = 0

    best_metric = 0.0
    best_steps = -1

    for epoch in range(num_train_epochs):
        for step, item in enumerate(train_dataloader):
            inputs = _prepare_input(item, device=args.device)
            outputs = model(**inputs, return_dict=True)
            loss = outputs.loss

            if args.gradient_accumulation_steps > 0:
                loss /= args.gradient_accumulation_steps
            
            loss.backward()
            t_loss += loss.detach()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()

                model.zero_grad()
                global_steps += 1

                if global_steps % args.logging_steps == 0:
                    print(f'Training: Epoch {epoch + 1}/{num_train_epochs} - Step {(step + 1) // args.gradient_accumulation_steps} - Loss {t_loss}')

                t_loss = 0.0

            if (global_steps + 1) % args.eval_steps == 0:
                
                loss, acc = evaluate(args, model, dev_dataloader)
                print(f'Evaluation: Epoch {epoch + 1}/{num_train_epochs} - Step {(global_steps + 1) // args.gradient_accumulation_steps} - Loss {loss} - Accuracy {acc}')

                if acc > best_metric:
                    best_metric = acc
                    best_steps = global_steps
                    
                    saved_dir = os.path.join(args.output_dir, f'checkpoint-{best_steps}')
                    os.makedirs(saved_dir, exist_ok=True)
                    model.save_pretrained(saved_dir)
                    tokenizer.save_vocabulary(save_directory=saved_dir)

    return best_steps, best_metric

def predict(
    args: TrainingArguments,
    model: PreTrainedModel,
    test_dataset,
    data_collator
):
    
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        # num_workers=args.dataloader_num_workers,
        collate_fn=data_collator
    )
    print("***** Running prediction *****")
    model.eval()
    preds_list = []

    for item in test_dataloader:
        inputs = _prepare_input(item, device=args.device)

        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)

            preds = torch.argmax(outputs.logits.cpu(), dim=-1).numpy()
            preds_list.append(preds)

    print(f'Prediction Finished!')
    preds = np.concatenate(preds_list, axis=0).tolist()

    model.train()

    return preds


def generate_commit(output_dir, task_name, test_dataset, preds: List[int]):

    test_examples = test_dataset.examples
    pred_test_examples = []
    for idx in range(len(test_examples)):
        example = test_examples[idx]
        label  = test_dataset.id2label[preds[idx]]
        pred_example = {'id': example.guid, 'query1': example.text_a, 'query2': example.text_b, 'label': label}
        pred_test_examples.append(pred_example)
    
    with open(os.path.join(output_dir, f'{task_name}_test.json'), 'w', encoding='utf-8') as f:
        json.dump(pred_test_examples, f, indent=2, ensure_ascii=False)


# %%
if __name__ == "__main__":
    import time
    data_args = DataTrainingArguments()
    training_args = TrainingArguments()
    print(data_args)
    print(training_args)

    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(data_args.model_dir)

    # initialize dataset
    processor = QQRProcessor(data_args.data_dir)
    train_dataset = ClassificationDataset(
        processor.get_train_examples(),
        label_list=processor.get_labels(),
        tokenizer=tokenizer,
        max_length=data_args.max_length,
    )
    dev_dataset = ClassificationDataset(
        processor.get_dev_examples(),
        label_list=processor.get_labels(),
        tokenizer=tokenizer,
        max_length=data_args.max_length,
    )
    test_dataset = ClassificationDataset(
        processor.get_test_examples(),
        label_list=processor.get_labels(),
        tokenizer=tokenizer,
        max_length=data_args.max_length,
    )

    data_collator = DefaultDataCollator()

    # %%
    model_name = f'{os.path.split(data_args.model_dir)[-1]}-{str(int(time.time()))}'
    training_args.output_dir = os.path.join(training_args.output_dir, model_name)
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)

    model = BertForSequenceClassification.from_pretrained(data_args.model_dir, num_labels=len(processor.get_labels()))
    model.to(training_args.device)

    best_steps, best_metric = train(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        data_collator=data_collator
    )

    print(f'Training Finished! Best step - {best_steps} - Best accuracy {best_metric}')

    best_model_dir = os.path.join(training_args.output_dir, f'checkpoint-{best_steps}')
    model = BertForSequenceClassification.from_pretrained(best_model_dir, num_labels=len(processor.get_labels()))
    model.to(training_args.device)

    model.save_pretrained(training_args.output_dir)
    torch.save(training_args, os.path.join(training_args.output_dir, 'training_args.bin'))
    tokenizer.save_vocabulary(save_directory=training_args.output_dir)

    # %%
    preds = predict(training_args, model, test_dataset, data_collator)
    generate_commit(training_args.output_dir, processor.TASK, test_dataset, preds)
