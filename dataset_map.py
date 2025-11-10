#!/usr/bin/env python
# coding=utf-8
"""
@author: xiaxu
@license: MIT
@file: dataset_map.py
@date: 2025/11/10 11:40
@desc: 
"""
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

# Dataset的使用与加载
model_name = 'bert-base-chinese'
model = AutoModel.from_pretrained(model_name) # 下载模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset_dict = load_dataset('csv',data_files={'data/raw/train.csv','data/raw/test.csv'}) #读取，返回datasetDict
dataset = dataset_dict['train'] #返回Dataset对象
print(dataset[0]['review']) #提取指定行（样本）
print(dataset[0:2]) #切片 {’cat‘：[a，b]，,label:[,],review:[,]}
dataset = dataset.remove_columns(['cat']) #删除指定字段
dataset = dataset.filter(lambda x: x['review'] is not None and x['review'].strip()!= '' and x['label'] in [0,1])  #过滤
dataset_dict = dataset.train_test_split(test_size=0.2)  #划分数据集和验证集
train_dataset = dataset_dict['train']
test_dataset = dataset_dict['test']
print(train_dataset[0])
def tokenize(text):
    encoded = tokenizer(text['review'], padding='max_length', truncation=True, max_length=10)#list
    text['input_ids'] = encoded['input_ids'] #编码后的输出，添加到新增字段
    text['attention_mask'] = encoded['attention_mask']
    return text
train_dataset = train_dataset.map(tokenize,batched=True) #map与tokenizer结合完成文本编码，完成分词、编码、打包成数据集
test_dataset = test_dataset.map(tokenize,batched=True)
print(train_dataset[0])
# 保存数据集
dataset_dict.save_to_disk('./data/processed') #保存为arrow格式;dataset_dict.json/
dataset_dict = load_from_disk('./data/processed') #加载
print(dataset_dict['train'][0])
train_dataset.to_csv('./data/processed/train.csv',index=False) #保存为csv,index = True会生成一个index column为空
test_dataset.to_csv('./data/processed/test.csv',index=False)
train_dataset.to_json('./data/processed/train.json',orient='records')
test_dataset.to_json('./data/processed/test.json',orient='records')
dataset_dict = load_dataset('csv',data_files={'data/processed/train.csv','data/processed/test.csv'})
print(dataset_dict['train'][0])
dataset_dict = load_dataset('json',data_files={'data/processed/train.json','data/processed/test.json'})
print(dataset_dict['train'][0])
#集成pytorch的Dataloader
train_dataset.set_format(
    type='torch',
    columns=['input_ids', 'attention_mask', 'label'],
) #将指定的列转换为指定的格式（tensor）,指定字段在访问时自动转换为tensor类型，底层数据存储不变，未指定字段在访问时将被自动过滤
train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True)
for batch in train_dataloader:
    print(batch.keys())
    print(batch['input_ids'].shape)
    break



