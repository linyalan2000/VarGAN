# merge all jsonl files from one dataset into one jsonl file
import jsonlines
import os
import pickle

def load_jsonl(file_path):
    with jsonlines.open(file_path) as reader:
        data = list(reader)
    return data

for mode in ['train', 'valid', 'test']:
    for language_name in ['java']:
        with jsonlines.open(f'data/{language_name}_{mode}.jsonl', mode='w') as writer:
            # calculate the number of files in the folder
            num_files = 0
            for file in os.listdir(f'/data/lyl/CodeSearchNet/original/{language_name}/final/jsonl/{mode}/'):
                if file.endswith('.jsonl'):
                    num_files += 1
            # merge all jsonl files into one jsonl file
            for file_idx in range(num_files):
                with jsonlines.open(f'/data/lyl/CodeSearchNet/original/{language_name}/final/jsonl/{mode}/{language_name}_{mode}_{file_idx}.jsonl') as reader:
                    for obj in reader:
                        writer.write((' '.join(obj['code_tokens'])).replace('\n', '').replace('\r', ''))

