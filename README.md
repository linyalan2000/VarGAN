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
The experiment based on VarCLR was improved based on [VarCLR](https://github.com/squaresLab/VarCLR) and the corresponding code is placed in the `VarCLR` folder. The training command for the model is:
```
python -m varclr.pretrain --model bert --name varclr-codebert --sp-model split --batch-size 64 --lr 1e-5 --epochs 1 --bert-model codebert --dis_lr 2e-5  --label new 
```

### test on IdBench
```
python -m varclr.pretrain --model bert --name varclr-codebert --sp-model split --batch-size 64 --lr 1e-5 --epochs 1 --bert-model codebert --dis_lr 2e-5  --label new --test  --load-file SAVED_MODEL_PAHT
```
### test on Variable Similarity Search task
```
cd varclr/utils
python infer_codebert.py
python similarity_search.py
```

## Deploying models on other downstream tasks

Both code summarization and clone detection code can be found in [CodeXGLUE](https://github.com/microsoft/CodeXGLUE/tree/main). Our model can be loaded in this way:

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