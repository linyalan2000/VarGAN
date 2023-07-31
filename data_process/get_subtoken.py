# get tokens from codes
import jsonlines
import json
import os
modes = ['train', 'valid', 'test']
token_map = {}
for mode in modes:
    with jsonlines.open(f'data/java_{mode}.jsonl') as reader:
        for obj in reader:
            for token in obj.split(' '):
                token = token.lower()
                if token not in token_map:
                    token_map[token] = 1
                else:
                    token_map[token] += 1
# sort the token_map by value
token_ls = sorted(token_map.items(), key=lambda x: x[1], reverse=True)
# change token_map to a dict
# token_map = {tok[0] : tok[1] for tok in token_ls}
with open('data/java_token_map.json', 'w') as f:
    json.dump(token_ls, f)