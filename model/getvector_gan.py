import torch
import json
from transformers import RobertaForMaskedLM,RobertaConfig, RobertaModel
from discremiter import RobertaClassificationHead
from tokenizer import Tokenizer
import numpy as np
import pickle
import random
from loadervectordata import VectorData

tokenizer = Tokenizer()
#model = RobertaForMaskedLM.from_pretrained('/home/lyl/VarCLR/codebert', from_pretrained=False)
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
        # #print(tokens)
        tokens=torch.tensor(tokens).unsqueeze(0)
        # #print(tokens)
        mask_attn=tokens.ne(50624).unsqueeze(0)
        #print(mask_attn)
        #sentence_model= model(input_ids=batch[0], attention_mask=batch[1],output_hidden_states=True)
        sentence_model= model(input_ids=tokens, attention_mask=mask_attn,output_hidden_states=True)
        #sentence_model=model(sentence)
        return sentence_model.hidden_states[0]
        #print(vector)
java_token_map_file=f'data/java_token_map.json'
java_token_map=json.load(open(java_token_map_file))
vector1=[]
vector=[]
name_frequency=[]
# f=open("vector.pkl","wb")
# f2=open("name_frequency.pkl","wb")
# f=open("vector_gan.pkl","wb")
# f2=open("name_frequency_gan.pkl","wb")

# for i in range(5000):
#      #a=random.randrange(0,50260)
#      vector=getvector(encoder,java_token_map[i][0]).squeeze()#torch.Size([1, 256, 768])
#      name_frequency.append(java_token_map[i])
#      vector1.append(vector[1])
#      if i%40==0 and not i==0:
#          print('  i {:>5,}  has got.'.format(i))
#      #result1=vector.detach().numpy().reshape(-1,20)
#      #np.savetxt("vector.csv",result1)
# for i in range(5000):
#      vector=getvector(encoder,java_token_map[i+45259][0]).squeeze()#torch.Size([1, 256, 768])
#      name_frequency.append(java_token_map[i+45259])
#      vector1.append(vector[1])
#      if i%40==0 and not i==0:
#         print('  i {:>5,}  has got.'.format(i))


# pickle.dump(vector1,f)
# f.close()
# pickle.dump(name_frequency,f2)
# f2.close()
# print('end')

with open("vector_gan_12.pkl","wb") as f:
     for i in range(5000):
          vector=getvector(encoder,java_token_map[i][0]).squeeze()#torch.Size([1, 256, 768])
          pickle.dump(vector[1],f)
          if i%40==0 and not i==0:
               print('  i {:>5,}  has got.'.format(i))
     for i in range(5000):
          vector=getvector(encoder,java_token_map[i+45259][0]).squeeze()#torch.Size([1, 256, 768])
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




