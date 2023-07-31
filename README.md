# VarGAN
This repository contains code for VarGAN, a novel method for variable name representations. It enhances the training of low-frequency variables through
adversarial training. 

## Requirments
```
torch == 1.11.0
spiral == 1.1.0
javalang == 0.13.0
```
## Pre-train

## Identifier-related tasks on VarCLR

## Deploying models on other downstream tasks
```
ckpt_path = 'gen.pkl'
tmp_dict = torch.load(ckpt_path)
encoder_dict = {}
for i in tmp_dict:
    ls = i.split('.')
    if ls[0] =='encoder':
        new_key = '.'.join(ls[1:])
        encoder_dict[new_key] = tmp_dict[i]
model.load_state_dict(encoder_dict)
```