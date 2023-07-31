
from dataclasses import replace
import imp
import torch 
import torch.utils.data as data
import torch.nn as nn
import pandas as pd
import pickle
import random
import numpy as np
import jsonlines
from transformers import RobertaTokenizer, T5Tokenizer, T5ForConditionalGeneration, BartTokenizer
use_cuda = torch.cuda.is_available()
tokenizer = RobertaTokenizer.from_pretrained("codebert-base") 
class BertData(data.Dataset):
    """
    Dataset that has binary samples.
    """
    def __init__(self, file_name):
        # 1. Initialize file path or list of file names.
        self.data = []
        with jsonlines.open(file_name, 'r') as reader:
            for item in reader:
                self.data.append(item)

    def __getitem__(self, offset):
        code_tokens = tokenizer.tokenize(self.data[offset]['mask_tokens'])
        new_ls = []
        for i in range(len(code_tokens)):
            if not (i != 0 and code_tokens[i] == '.' and code_tokens[i - 1] == '<mask>'):
                new_ls.append(code_tokens[i])
        code = tokenizer.encode(new_ls, return_tensors="pt", padding="max_length",
        truncation=True,
        max_length=256)
        code_input = code[0]
        # code_attn = code['attention_mask'][0]
        org_code_str = self.data[offset]['code']
        code_lable = tokenizer(org_code_str, return_tensors="pt", padding="max_length",
        truncation=True,
        max_length=256)['input_ids'][0]
        return code_input, self.data[offset]['label'], code_lable

    def __len__(self):
        return len(self.data)

class BertTokenData(data.Dataset):
    """
    Dataset that has binary samples.
    """
    def __init__(self, file_name):
        # 1. Initialize file path or list of file names.
        self.data = []
        with jsonlines.open(file_name, 'r') as reader:
            for item in reader:
                self.data.append(item)

    def __getitem__(self, offset):
        code_tokens = tokenizer.tokenize(self.data[offset]['mask_tokens'])
        new_ls = []
        for i in range(len(code_tokens)):
            if not (i != 0 and code_tokens[i] == '.' and code_tokens[i - 1] == '<mask>'):
                new_ls.append(code_tokens[i])
        code = tokenizer.encode(new_ls, return_tensors="pt", padding="max_length",
        truncation=True,
        max_length=256)
        code_input = code[0]
        # code_attn = code['attention_mask'][0]
        org_code_str = self.data[offset]['code']
        code_lable = tokenizer(org_code_str, return_tensors="pt", padding="max_length",
        truncation=True,
        max_length=256)['input_ids'][0]
        token_label = torch.tensor(self.data[offset]['token_label'])
        fake_token_label = torch.tensor(self.data[offset]['fake_token_label'])
        return code_input, self.data[offset]['label'], code_lable, token_label, fake_token_label

    def __len__(self):
        return len(self.data)

class BertRandomData(data.Dataset):
    """
    Dataset that has binary samples.
    """
    def __init__(self, file_name):
        # 1. Initialize file path or list of file names.
        self.data = []
        self.threshhold = 423
        
        with jsonlines.open(file_name, 'r') as reader:
            for item in reader:
                self.data.append(item)

    def __getitem__(self, offset):
        code_tokens = tokenizer.tokenize(self.data[offset]['mask_tokens'])
        new_ls = []
        for i in range(len(code_tokens)):
            if not (i != 0 and code_tokens[i] == '.' and code_tokens[i - 1] == '<mask>'):
                new_ls.append(code_tokens[i])
        code = tokenizer.encode(new_ls, return_tensors="pt", padding="max_length",
        truncation=True,
        max_length=256)
        code_input = code[0]
        # code_attn = code['attention_mask'][0]
        org_code_str = self.data[offset]['code']
        code_lable = tokenizer(org_code_str, return_tensors="pt", padding="max_length",
        truncation=True,
        max_length=256)['input_ids'][0]
        return code_input, self.data[offset]['label'], code_lable

    def __len__(self):
        return len(self.data)
