import pickle
import random
from spiral import safe_simple_split
from transformers import RobertaTokenizer
from tree_sitter import Language, Parser
from tree_sitter import Language, Parser
tokenizer = RobertaTokenizer.from_pretrained('codebert-base')
path = '../contrast_learning/'
Language.build_library(
    # Store the library in the `build` directory
    'build/my-languages.so',

    # Include one or more languages
    [
        f'{path}/tree-sitter/tree-sitter-python',
        f'{path}/tree-sitter/tree-sitter-java',
        f'{path}/tree-sitter/tree-sitter-php',
        f'{path}/tree-sitter/tree-sitter-go',
        f'{path}/tree-sitter/tree-sitter-ruby',
        f'{path}/tree-sitter/tree-sitter-javascript',
    ]
)

JAVA_LANGUAGE = Language('build/my-languages.so', 'java')
PYTHON_LANGUAGE = Language('build/my-languages.so', 'python')
PHP_LANGUAGE = Language('build/my-languages.so', 'php')
GO_LANGUAGE = Language('build/my-languages.so', 'go')
RUBY_LANGUAGE = Language('build/my-languages.so', 'ruby')
JAVASCRIPT_LANGUAGE = Language('build/my-languages.so', 'javascript')
# map from language to tree-sitter language
LANGUAGE_MAP = {
    'java': JAVA_LANGUAGE,
    'python': PYTHON_LANGUAGE,
    'php': PHP_LANGUAGE,
    'go': GO_LANGUAGE,
    'ruby': RUBY_LANGUAGE,
    'javascript': JAVASCRIPT_LANGUAGE,
}
parser = Parser()

def getCodeFromFiles(path):
    """ Return a list of code lines from a pkl file.
    """
    cnt = 0
    codes = []
    with open(path, 'rb') as f:
        while True:
            try:
                code = pickle.load(f)
                codes.append(code)
                cnt += 1
                if cnt == 10000:
                    break
            except EOFError:
                break
    return codes

def isSeldomWord(name, freq_map, threshold=500):
    name_ls = safe_simple_split(name)
    for sub in name_ls:
        if sub.lower() not in freq_map or freq_map[sub.lower()] < threshold:
            return True
    return False
# get the identifier from the code with the fix position
def get_identifier_from_position(code_string, start_point, end_point):
    lines = code_string.splitlines()
    identifier = lines[start_point[0]][start_point[1]:end_point[1]]
    return identifier

# get the identifier from the code 
def get_identifier(code, lang):
    pos = []
    identifiers = []
    def traverse(root):
        if root is None:
            return
        for child in root.children:
            if child.type == 'identifier':
                start_point = child.start_point
                end_point = child.end_point
                pos.insert(0, (start_point, end_point))
            traverse(child)
    parser.set_language(LANGUAGE_MAP[lang])
    tree = parser.parse(bytes(code, 'utf-8'))
    traverse(tree.root_node)
    for id in pos:
        identifiers.append(get_identifier_from_position(code, id[0], id[1]))
    # return identifiers, and the position of all identifiers
    return list(set(identifiers)), pos


# get the identifier from the code 
def get_identifier_subset(code, lang):
    pos = []
    identifiers = []
    def get_id_from_block(block):
        if block is None:
            return
        for child in block.children:
            if child.type == 'identifier':
                start_point = child.start_point
                end_point = child.end_point
                pos.insert(0, (start_point, end_point))
            traverse(child)
    def traverse(root):
        if root is None:
            return
        for child in root.children:
            if child.type == 'variable_declarator' or 'formal_parameter':
                get_id_from_block(child)
            traverse(child)
    parser.set_language(LANGUAGE_MAP[lang])
    tree = parser.parse(bytes(code, 'utf-8'))
    traverse(tree.root_node)
    for id in pos:
        identifiers.append(get_identifier_from_position(code, id[0], id[1]))
    # return identifiers, and the position of all identifiers
    return list(set(identifiers)), pos

# given a threshold, return the high frequency words number minus low frequency words number
def get_high_low_freq_words_num(code_ls, lang, freq_map, threshold, p):
    high_freq_words_num = 0
    low_freq_words_num = 0
    for code in code_ls:
        code_item, code_tokens = code.split('****\t****')
        identifiers, _ = get_identifier(code_item, lang)
        flag = 1
        mask_name = []
        for name in identifiers:
                k = random.random()
                if k < p:
                    mask_name.append(name)
        for identifier in identifiers:
            if identifier not in mask_name and isSeldomWord(identifier, freq_map, threshold):
                low_freq_words_num += 1
                flag = 0
                break
        high_freq_words_num += flag
    return high_freq_words_num - low_freq_words_num

def get_high_low_freq_token_num(code_ls, lang, freq_map, threshold, p):
    high_freq_words_num = 0
    low_freq_words_num = 0
    for code in code_ls:
        code_item, code_tokens = code.split('****\t****')
        identifiers, _ = get_identifier(code_item, lang)
        mask_name = []
        for name in identifiers:
                k = random.random()
                if k < p:
                    mask_name.append(name)
        for identifier in identifiers:
            if identifier not in mask_name:
                if isSeldomWord(identifier, freq_map, threshold):
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
def find_threshold(code_ls, lang, freq_map, p):
    """
    find the threshold of high frequency words
    """
    def f(threshold):
        return get_high_low_freq_words_num(code_ls, lang, freq_map, threshold, p)
    return find_zero(f, 50, 10000)

def find_threshold_token(code_ls, lang, freq_map, p):
    """
    find the threshold of high frequency words
    """
    def f(threshold):
        return get_high_low_freq_token_num(code_ls, lang, freq_map, threshold, p)
    return find_zero(f, 4000, 10000)

def get_mask_token_for_identifier(identifier, freq_map, threshold, tokenizer):
    length = len(tokenizer.encode(' ' + identifier)) - 2
    # print(tokenizer.encode(identifier))
    is_seldom = isSeldomWord(identifier, freq_map, threshold)
    if is_seldom:
        return ' 1' * length + ' '
    else:
        return ' 0' * length + ' '
def get_token_lable(code, identifiers, mask_name, freq_map, threshold):
    for i in range(len(identifiers)):
        if identifiers[i] not in mask_name:
            mask_id = get_mask_token_for_identifier(identifiers[i], freq_map, threshold, tokenizer)
            code = code.replace(' ' + identifiers[i] + ' ', mask_id)
    token_id = tokenizer.tokenize(code)
    token_ls = [-100] * 256
    fake_token_ls = [-100] * 256
    for i in range(min(256, len(token_id))):
        if not(token_id[i] != 'Ġ0' and token_id[i] != 'Ġ1'):
            token_ls[i] = int(token_id[i][-1])
            fake_token_ls[i] = 1 - int(token_id[i][-1])
    return token_ls, fake_token_ls


