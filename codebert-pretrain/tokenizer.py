import json
class Tokenizer():
    def __init__(self) -> None:
        with open('data/java_token_map.json', 'r') as f:
            self.vocab = json.load(f)
        self.vocab_ls = self.vocab[:50260]
        self.vocab = {token[0]: idx for idx, token in enumerate(self.vocab_ls)}
        self.vocab['<unk>'] = len(self.vocab)
        self.vocab['<sos>'] = len(self.vocab)
        self.vocab['<eos>'] = len(self.vocab)
        self.vocab['<mask>'] = len(self.vocab)
        self.vocab['<pad>'] = len(self.vocab)
        # Generate a reverse glossary
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}

    def tokenize(self, code, max_len=256):
        code='<sos> ' + code + ' <eos>'
        tokens = code.split(' ')
        tokens = [token.lower() for token in tokens]
        tokens = [self.vocab[token] if token in self.vocab else self.vocab['<unk>'] for token in tokens]
        # if length of tokens is less than max_len, pad it with <pad>
        if len(tokens) < max_len:
            tokens += [self.vocab['<pad>']] * (max_len - len(tokens))
        # if length of tokens is greater than max_len, truncate it
        tokens = tokens[:max_len]
        return tokens
    
    def detokenize(self, tokens):
        tokens_ls = [ids.item() for ids in tokens]
        return ' '.join([self.reverse_vocab[token] for token in tokens_ls])
    
    def __len__(self):
        return len(self.vocab)
    
    def __getitem__(self, idx):
        return self.vocab_ls[idx]
    
    def __contains__(self, token):
        return token in self.vocab
