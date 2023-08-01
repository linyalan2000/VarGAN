# make dataset for VarGAN pretrain
# 1: low-frequency word  0: no low-frenquency words
from model.utils.identifiers_process import *
import json
from random import sample
import jsonlines
import argparse
def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument("--output_dir",type=str,default=".", help='output dir of result')
    parser.add_argument("--language_name",type=str,default="java", help='output dir of result')
    return parser
def main(params):  
    lang = params.language_name
    freq_map_file = f'subtoken_{lang}.json'
    freq_map = json.load(open(freq_map_file, 'r'))
    for mode in ['train', 'valid']:
        codes = getCodeFromFiles(f'{lang}_{mode}.pkl')[0]
        if mode == 'train':
            thresh = find_threshold(sample(codes, 3000), lang, freq_map)
        writer = jsonlines.open(f'data/{lang}_{mode}_data.jsonl', 'w')
        cnt = 0
        high_num = 0
        low_num = 0
        for item in codes:
            code_item, code_tokens = item.split('****\t****')
            varnames, _ = get_identifier(code_item, lang)
            tmp = {'code': code_tokens}
            for name in list(varnames):
                if isSeldomWord(name, freq_map, thresh):
                    tmp['label'] = 1
                    low_num += 1
                    break
            if 'label' not in tmp:
                tmp['label'] = 0 # no sel
                high_num += 1
            writer.write(tmp)
            cnt += 1
        writer.close()
        print(thresh)
        print(high_num)
        print(low_num)


    
if __name__ == '__main__':
    # get some parameters from user
    parser = get_parser()
    params = parser.parse_args()
    main(params)