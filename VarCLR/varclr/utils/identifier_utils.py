def getCodeFromFiles(path):
    """ Return a list of code lines from a pkl file.
    """
    cnt = 0
    codes = []
    with open(path, 'r') as f:
        for i in f:
            codes.append(i.strip())
    return codes

def isSeldomWord(name, freq_map, threshold=500):
    name_ls = name.split(' ')
    for sub in name_ls:
        if sub.lower() not in freq_map or freq_map[sub.lower()] < threshold:
            return '\t1'
    return '\t0'


# given a threshold, return the high frequency words number minus low frequency words number
def get_high_low_freq_words_num(code_ls, lang, freq_map, threshold):
    high_freq_words_num = 0
    low_freq_words_num = 0
    for code in code_ls:
        if '\t' not in code:
            continue
        var1, var2 = code.split('\t')
        if isSeldomWord(var1, freq_map, threshold) == '\t1':
            low_freq_words_num += 1
        else:
            high_freq_words_num += 1
        if isSeldomWord(var2, freq_map, threshold) == '\t1':
            low_freq_words_num += 1
        else:
            high_freq_words_num += 1
    return high_freq_words_num - low_freq_words_num


# find zero point of a function
def find_zero(f, a, b):
    """
    use bisection method to find the zero point of a function
    """
    while b - a > 10:
        c = int((a + b) / 2)
        if f(c) * f(a) <= 0:
            b = c
        else:
            a = c
    return a

# find the threshold of high frequency words
def find_threshold(code_ls, lang, freq_map):
    """
    find the threshold of high frequency words
    """
    def f(threshold):
        return get_high_low_freq_words_num(code_ls, lang, freq_map, threshold)
    return find_zero(f, 10, 10000)



def get_mask_token_for_identifier(identifier, freq_map, threshold, tokenizer):
    length = len(tokenizer.encode(' ' + identifier)) - 2
    # print(tokenizer.encode(identifier))
    is_seldom = isSeldomWord(identifier, freq_map, threshold)
    if is_seldom:
        return ' 1' * length + ' '
    else:
        return ' 0' * length + ' '