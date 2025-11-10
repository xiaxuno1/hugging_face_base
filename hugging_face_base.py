#!/usr/bin/env python
# coding=utf-8
"""
@author: xiaxu
@license: MIT
@file: hugging_face_base.py
@date: 2025/11/10 09:38
@desc: hf的用法解析
"""
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

# AutoModel类，用于下载预训练模型
model = AutoModel.from_pretrained('google-bert/bert-base-chinese') # 下载模型

model = AutoModel.from_pretrained('./pretrained/bert-base-chinese') #加载本地模型
print(model)

model = AutoModelForSequenceClassification.from_pretrained('google-bert/bert-base-chinese',
                                                           num_labels = 3) #下载具有特定功能的模型，文本分类，指定分类为3
print(type(model)) #<class 'transformers.models.bert.modeling_bert.BertForSequenceClassification'>
print(model)

# Tokenizer加载，分词、ID、token、padding
tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese') #加载Tokenizer

tokens = tokenizer.tokenize('我爱机器学习！') #分词
print(tokens) #['我', '爱', '机', '器', '学', '习', '！'] ->list[string]
ids = tokenizer.convert_tokens_to_ids(tokens) # 转id
print(ids)
tokens = tokenizer.convert_ids_to_tokens(ids) #id转tokens
print(tokens)
ids = tokenizer.encode('我爱机器学习',add_special_tokens=True) #编码，添加[cls][sep] padding
print(ids) #[101, 2769, 4263, 3322, 1690, 2110, 739, 102] 添加了[cls] [sep]
string = tokenizer.decode(ids,skip_special_tokens=False) #解码，
print(string)

text = ["我爱自然语言处理", "我爱人工智能", "我爱你"]
inputs = tokenizer(text) # dict:input_ids，token_type_ids（sep的前后标志0,1），attention_mask
print(inputs) #返回的为字典，字典内容为list
inputs = tokenizer(
    text,
    padding='max_length', # 填充
    truncation=True,    # 自动截断
    max_length=10,      # 统一最大长度
    return_tensors="pt"
) #转换为tensor
# 预训练模型流程
model_name = 'bert-base-chinese'
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = ["我爱自然语言处理", "我爱人工智能", "我爱你"]
encoded = tokenizer(text, padding='max_length', truncation=True, max_length=10, return_tensors="pt")
with torch.no_grad():
    outputs = model(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'],token_type_ids=encoded['token_type_ids'])
print(outputs.keys()) #输出last_hidden_state、pooler_outputs
print('last_hidden_state:',outputs.last_hidden_state.shape) #[B,S,H]
print('pooler_outputs:',outputs.pooler_output.shape) #执行完成后[cls]的结果[B,H]用于分类
# Dataset的使用与加载
dataset_dict = load_dataset('csv',data_files={'train.csv','test.csv'}) #读取，返回datasetDict
dataset = dataset_dict['train'] #返回Dataset对象
print(dataset[0]['review']) #提取指定行（样本）
print(dataset[0:2]) #切片
dataset = dataset.remove_columns(['cat']) #删除指定字段
dataset = dataset.filter(lambda x: x['review'] is not None and x['review'].strip()!= '' and x['label'] in [0,1])  #过滤
dataset_dict = dataset.train_test_split(test_size=0.2)  #划分数据集和验证集
train_dataset = dataset_dict['train']
test_dataset = dataset_dict['test']
print(train_dataset[0])
def tokenize(text):
    encoded = tokenizer(text, padding='max_length', truncation=True, max_length=10)#list
    text['input_ids'] = encoded['input_ids'] #编码后的输出，添加到新增字段
    text['attention_mask'] = encoded['attention_mask']
    return text
train_dataset = train_dataset.map(tokenize,batched=True) #map与tokenizer结合完成文本编码，完成分词、编码、打包成数据集
test_dataset = test_dataset.map(tokenize,batched=True)
print(train_dataset[0])



