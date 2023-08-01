'''
get the hidden states of each word in the vocabulary
'''

import torch
import json
from transformers import RobertaForMaskedLM,RobertaConfig, RobertaModel
from discremiter import RobertaClassificationHead
from tokenizer import Tokenizer
import pickle

tokenizer = Tokenizer()
config = RobertaConfig()
config.vocab_size = 50265
model = RobertaForMaskedLM(config)
encoder = RobertaModel(config)
classifier = RobertaClassificationHead(config)
ckpt_path = '/data2/lyl/codeGan_model/tmp_gen_1.pkl'
pretrained_dict = torch.load(ckpt_path)
classifier_dict = {}
new_dict = {}
pretrained_dict = {k: v for k, v in pretrained_dict.items()}
for i in pretrained_dict:
    key_words = i.split('.')
    if key_words[0] == 'encoder':
        new_dict['.'.join(key_words[1:])] = pretrained_dict[i]
    if key_words[0] == 'classifier':
        classifier_dict['.'.join(key_words[1:])] = pretrained_dict[i]
encoder.load_state_dict(new_dict)


def getvector(model,sentence):
        tokens=tokenizer.tokenize(sentence)
        tokens=torch.tensor(tokens).unsqueeze(0)
        mask_attn=tokens.ne(50624).unsqueeze(0)
        sentence_model= model(input_ids=tokens, attention_mask=mask_attn,output_hidden_states=True)
        return sentence_model.hidden_states[0]

java_token_map_file=f'data/java_token_map.json'
java_token_map=json.load(open(java_token_map_file))
vector1=[]
vector=[]
name_frequency=[]

# get the data and write to a file
with open("vector_gan_12.pkl","wb") as f:
     for i in range(5000):
          vector=getvector(encoder,java_token_map[i][0]).squeeze() #the vector with 768 dimensions is needed
          pickle.dump(vector[1],f)
          if i%40==0 and not i==0:
               print('  i {:>5,}  has got.'.format(i))
     for i in range(5000):
          vector=getvector(encoder,java_token_map[i+45259][0]).squeeze()
          pickle.dump(vector[1],f)
          if i%40==0 and not i==0:
               print('  i {:>5,}  has got.'.format(i))

f.close()
with open("name_frequency_gan_12.pkl","wb") as f:
     for i in range(5000):
          name_frequency=java_token_map[i]
          pickle.dump(name_frequency,f)
          if i%40==0 and not i==0:
               print('  i {:>5,}  has got.'.format(i))
     for i in range(5000):
          name_frequency=java_token_map[i+45259]
          pickle.dump(name_frequency,f)
          if i%40==0 and not i==0:
               print('  i {:>5,}  has got.'.format(i))

f.close()

print('end')




