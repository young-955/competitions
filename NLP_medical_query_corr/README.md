# NLP_medical_query_corr
阿里天池NLP医疗query相似度学习赛
https://tianchi.aliyun.com/competition/entrance/532001/information

## Transformer算法的复现
attention_reproduction.py
    一.复现baseline
    1.文中word2vec数据库不知道是哪一个，没有提供，从网上下载一个阿里的数据库
    2.调整数据结构，函数逻辑
    3.清理数据，评估数据中包含标签为“NA”的数据
    4.跑出一个baseline模型和结果
    5.预测代码有bug，进行预测时还需要标签，修改bug
    复现模型在测试集上的结果：76.88
    二.模型调优
    1.调整epoch，从27增加到54

## BERT算法的复现
bert_reproduction.py
    一.复现baseline
    调整结构，解决bug
    复现模型在测试集上的结果：82.71

## 使用集成学习
### bagging
简单投票法：多个模型投票，取高票数结果

### boosting
互相学习法：

### stacking
领导开会决策法：每个模型提供结果，输入给决策模型，决策模型根据这些输出最终结果
