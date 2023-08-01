import torch 
import torch.utils.data as data
import jsonlines
from transformers import RobertaTokenizer
import random
use_cuda = torch.cuda.is_available()
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base") 
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

class BertMyTokData(data.Dataset):
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
        code_tokens = tokenizer.tokenize(self.data[offset])
        # change some to <mask>
        # count the number of code_tokens that not equal to 50264
        code_len = len([token for token in code_tokens if token != 50264])
        mask_code_tokens = code_tokens.copy()
        mask_num = int((code_len - 2) * 0.15)
        mask_idx = random.sample(range(1, code_len - 1), mask_num)
        for idx in mask_idx:
            mask_code_tokens[idx] = 50263
        return torch.tensor(mask_code_tokens), torch.tensor(code_tokens)

    def __len__(self):
        return len(self.data)
    use_cuda = torch.cuda.is_available()

class BertMyTokGanData(data.Dataset):
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
        code_tokens = self.data[offset]['mask_tokens']
        new_ls = []
        for i in range(len(code_tokens)):
            if not (i != 0 and code_tokens[i] == '.' and code_tokens[i - 1] == '<mask>'):
                new_ls.append(code_tokens[i])
        code_input=torch.tensor(tokenizer.tokenize(' '.join(new_ls)))
        org_code_str = self.data[offset]['code']
        lable=tokenizer.tokenize(org_code_str)
        code_lable=torch.tensor(lable)
        return code_input, self.data[offset]['label'], code_lable

    def __len__(self):
        return len(self.data)
