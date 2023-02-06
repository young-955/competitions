# %%
from bert_reproduction import *
from attention_reproduction import *

# word2vec模型路径
w2v_file = r'data\tencent-ailab-embedding-zh-d100-v0.2.0-s.txt'
#载入词向量
w2v_model = KeyedVectors.load_word2vec_format(w2v_file, binary=False)

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

    model.to(training_args.device)
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
    # 数据路径
    data_dir = r'data'

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
            preds = torch.argmax(outputs[1].cpu(), dim=-1).numpy()
            preds_list.append(preds)

    preds = np.concatenate(preds_list, axis=0).tolist()

    return preds

def generate_commit(output_dir, task_name, test_dataset, preds: List[int]):
    test_examples = test_dataset.examples
    pred_test_examples = []
    for idx in range(len(test_examples)):
        example = test_examples[idx]
        print(preds)
        label  = test_dataset.id2label[preds[idx]]
        pred_example = {'id': example.guid, 'query1': example.text_a, 'query2': example.text_b, 'label': label}
        pred_test_examples.append(pred_example)
    
    with open(os.path.join(output_dir, f'{task_name}_test.json'), 'w', encoding='utf-8') as f:
        json.dump(pred_test_examples, f, indent=2, ensure_ascii=False)

# %%
# bagging
def bagging(models, funcs, test=False):
    preds = []
    for model, func in zip(models, funcs):
        preds.append(func(model, test))
        torch.cuda.empty_cache()

    # vote decision
    res = []
    for i in range(len(preds[0])):
        per_pred = np.array([0, 0, 0])
        for p in np.array(preds)[:, i]:
            per_pred[p] += 1
        r = np.where(per_pred == max(per_pred))
        if len(r) > 1:
            r = [preds[0, i]]
        
        res.append(r[0][0])

    data_dir = r'data'
    processor = QQRProcessor(data_dir=data_dir)
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
    generate_commit('res', 'bagging', dataset, res)

#  计算准确率
def cal_acc(file):    
    data_dir = r'data'
    processor = QQRProcessor(data_dir=data_dir)
    data_collator = DataCollator()
    eval_dataset = QQRDataset(
        processor.get_dev_examples(),
        processor.get_labels(),
        vocab_mapping=w2v_model.key_to_index,
        max_length=32
    )
    pred_examples = QQRDataset(
        processor._create_examples(file),
        processor.get_labels(),
        vocab_mapping=w2v_model.key_to_index,
        max_length=32
    )
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=data_collator
    )
    pred_dataloader = DataLoader(
        dataset=pred_examples,
        batch_size=32,
        shuffle=False,
        collate_fn=data_collator
    )

    preds_list = []
    labels_list = []

    for item in eval_dataloader:
        inputs = _prepare_input(item)
        labels_list.append(inputs['labels'].cpu().numpy())
    
    for item in pred_dataloader: 
        inputs = _prepare_input(item)
        preds_list.append(inputs['labels'].cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    accuracy = simple_accuracy(preds, labels)
    print(f'accuracy: {accuracy}')

    return accuracy

# %%
def main():
    bert_model_path = r'model\chinese-bert-wwm-ext-1663495751'
    attention_model_1_path = r'model\attention_20220914_score7688.pth'
    attention_model_2_path = r'model\attention_20220918_score7563.pth'

    # load model
    bert_model = BertForSequenceClassification.from_pretrained(bert_model_path, num_labels=3)
    attention_model_1 = torch.load(attention_model_1_path)
    attention_model_2 = torch.load(attention_model_2_path)

    models = [bert_model, attention_model_1, attention_model_2]
    funcs = [bert_func, attention_func, attention_func]
    # models = [attention_model_2]
    # funcs = [attention_func]
    bagging(models, funcs, False)


# %%
if __name__ == "__main__":
    # main()
    cal_acc('res/bagging_test.json')
    
# %%
