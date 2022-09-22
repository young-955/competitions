# %%
from bert_reproduction import *
from attention_reproduction import *


# %%
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

# bert model predict func
def bert_func(model, test=False):
    data_args = DataTrainingArguments()
    training_args = TrainingArguments()
    tokenizer = AutoTokenizer.from_pretrained(data_args.model_dir)
    processor = QQRProcessor(data_args.data_dir)
    data_collator = DefaultDataCollator()

    data_examples = None
    if test:
        data_examples = processor.get_test_examples()
    else:
        data_examples = processor.get_dev_examples()

    # prepare data
    dataset = ClassificationDataset(
        data_examples,
        label_list=processor.get_labels(),
        tokenizer=tokenizer,
        max_length=data_args.max_length,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=training_args.eval_batch_size,
        shuffle=False,
        # num_workers=args.dataloader_num_workers,
        collate_fn=data_collator
    )

    model.to(device)
    # predict
    model.eval()
    preds_list = []
    for item in dataloader:
        inputs = _prepare_input(item, device=training_args.device)

        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
            preds = torch.argmax(outputs.logits.cpu(), dim=-1).numpy()
            preds_list.append(preds)

    preds = np.concatenate(preds_list, axis=0).tolist()

    return preds

# attention model predict func
def attention_func(model, test=False):
    # 参数
    eval_batch_size = 32
    device = 'cuda'
    # word2vec模型路径
    w2v_file = r'data\tencent-ailab-embedding-zh-d100-v0.2.0-s\tencent-ailab-embedding-zh-d100-v0.2.0-s.txt'
    # 数据路径
    data_dir = r'data\tencent-ailab-embedding-zh-d100-v0.2.0-s'

    #载入词向量
    w2v_model = KeyedVectors.load_word2vec_format(w2v_file, binary=False)
    processor = QQRProcessor(data_dir=data_dir)
    data_collator = DataCollator()

    data_examples = None
    if test:
        data_examples = processor.get_test_examples()
    else:
        data_examples = processor.get_dev_examples()

    dataset = QQRDataset(
        data_examples,
        processor.get_labels(),
        vocab_mapping=w2v_model.key_to_index,
        max_length=32
    )
    
    # predict
    model.to(device)
    model.eval()
    preds_list = []
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=data_collator
    )
    
    for item in dataloader:
        inputs = _prepare_input(item, device=device)

        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits.cpu(), dim=-1).numpy()
            preds_list.append(preds)

    preds = np.concatenate(preds_list, axis=0).tolist()

    return preds

# %%
# bagging
def bagging(models, funcs):
    preds = []
    for model, func in zip(models, funcs):
        preds.append(func(model))

    # vote decision
    res = []
    for p in preds:
        p

# %%
if __name__ == "__main__":
    bert_model_path = r'model\chinese-bert-wwm-ext-1663495751'
    attention_model_1_path = r'model\attention_20220914_score7688.pth'
    attention_model_2_path = r'model\attention_20220918_score7563.pth'

    # %%
    # load model
    bert_model = BertForSequenceClassification.from_pretrained(best_model_dir, num_labels=len(processor.get_labels()))
    attention_model_1 = torch.load(attention_model_1_path)
    attention_model_1 = torch.load(attention_model_2_path)
