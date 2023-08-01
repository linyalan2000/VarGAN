# 代码 + 1表示有低频词 0表示没有低频词
from varclr.utils.identifier_utils import *
import json
from random import sample
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
def get_data(thresh):  
    freq_map_file = f'subtoken_js.json'
    # freq_map_file = f'../contrast_learning/subtoken_{lang}.json'
    freq_map = json.load(open(freq_map_file, 'r'))
    codes = getCodeFromFiles(f'cs-cs.var.tok.txt')
    writer = open(f'cs-cs-freq-self_bak.var.tok.txt', 'w')
    cnt = 0
    high_num = 0
    low_num = 0
    for item in codes:
        if '\t' not in item:
            cnt += 1
            continue
        var1, var2 = item.split('\t')
        writer.write(var1 + '\t' + var2 + isSeldomWord(var1, freq_map, thresh) + isSeldomWord(var2, freq_map, thresh) + '\n')
        if isSeldomWord(var1, freq_map, thresh) == '\t1':
            low_num += 1
        else:
            high_num += 1
        if isSeldomWord(var2, freq_map, thresh) == '\t1':
            low_num += 1
        else:
            high_num += 1

    writer.close()
    print(thresh)
    print(high_num)
    print(low_num)
    print(cnt)
    return thresh

    
if __name__ == '__main__':
    # get some parameters from user
    parser = get_parser()
    params = parser.parse_args()
    get_data(61)