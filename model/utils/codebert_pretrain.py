'''
train the pretrained model without GAN
'''
import torch
from transformers import RobertaConfig, RobertaForMaskedLM
import numpy as np
from utils.tokenizer import Tokenizer
import time
import datetime
import random
import os
from utils.data_loader import BertMyTokData
from torch import nn
torch.cuda.set_device(1) 
tokenizer = Tokenizer()


#initialize BERT configuration
config = RobertaConfig()
config.vocab_size = 50265
model = RobertaForMaskedLM(config)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.train().to(device)
optimizer = torch.optim.AdamW(model.parameters(),
                  lr = 1e-5, # args.learning_rate - default is 5e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8
                )
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))
seed_val = 114
def save_model(model, epoch, timestamp):
    """Save model parameters to checkpoint"""
    os.makedirs(f'./save_model/', exist_ok=True)
    ckpt_path=f'./save_model/codebert_{epoch}.pkl'
    print(f'Saving model parameters to {ckpt_path}')
    torch.save(model.state_dict(), ckpt_path)

loss_fn = nn.CrossEntropyLoss()
softmax = nn.Softmax(dim = 1)
timestamp=datetime.datetime.now().strftime('%Y%m%d%H%M')

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
# 2~4
epochs = 3


# ensure a certain output when running the code
###############################################################################
# Load data
###############################################################################
train_set=BertMyTokData('data/java_train.jsonl')
valid_set=BertMyTokData('data/java_valid.jsonl')
train_loader=torch.utils.data.DataLoader(dataset=train_set, batch_size=16, shuffle=True, num_workers=1)
valid_loader=torch.utils.data.DataLoader(dataset=valid_set, batch_size=32, shuffle=False, num_workers=1)
print("Loaded data!")

total_t0 = time.time()
best_acc = 0
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    model.train()
    # time for a single epoch
    t0 = time.time()

    # reset total loss for each epoch
    total_train_loss = 0

    for step, batch in enumerate(train_loader):
        # output the progress information
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}. Loss = {:}  Elapsed: {:}.'.format(step, len(train_loader), total_train_loss / 40, elapsed))
            total_train_loss = 0
 
        # data
        code_input = batch[0].to(device)
        mask_attn = batch[0].ne(50264).to(device)
        label = batch[1].to(device)

        # reset grad 
        output= model(input_ids=code_input, attention_mask=mask_attn, output_hidden_states=True,labels=label)
        loss=output[0]
        
        # total loss
        total_train_loss += loss.item()
        
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()          
    # time for a single epoach
    training_time = format_time(time.time() - t0)

    print("")
    print("  Training epcoh took: {:}".format(training_time))

    
    # ========================================
    #               Validation
    # ========================================
    # after each epcoh
    result = open(f'data/valid_result_{epoch_i}_bad_name.txt', 'w')
    cnt = 0
    print("")
    print("Running Validation...")
    
    t0 = time.time()
    
    model.eval()
    
    # Tracking variables 
    total_eval_accuracy = 0
    total_name_ok_acc = 0
    total_eval_loss = 0
    predict_accuracy=0

    # Evaluate data for one epoch
    cnt = 0
    for batch in valid_loader:
        cnt += 1
        #data
        code_input = batch[0].to(device)
        code_attn = batch[0].ne(50264).to(device)
        label = batch[1].to(device)
        # reset grad 
        with torch.no_grad():    
            outputs = model(input_ids=code_input, attention_mask=code_attn, labels=label)
            logits = outputs.logits
            loss = outputs.loss
        predict_acc_cnt=0
        acc_cnt = 0
        total_cnt = 0
        avg_predict_accuracy=0

        _,idx=torch.max(logits,dim=2)
        for batch_idx in range(0, len(code_input)):
            for k in range(0,len(idx[batch_idx])):
                if code_input[batch_idx][k] == 50263: #'50623' is the padding
                    total_cnt += 1
                    if idx[batch_idx][k]== label[batch_idx][k]: # the prediction is correct
                        predict_acc_cnt += 1
                
        predict_accuracy += predict_acc_cnt / total_cnt

    # print accuracy for this epoch
    avg_predict_accuracy = predict_accuracy / len(valid_loader)
    print("  Predict Accuracy: {0:.2f}".format(avg_predict_accuracy))   
    validation_time = format_time(time.time() - t0)
    print("  Validation took: {:}".format(validation_time))
    #save model
    if avg_predict_accuracy > best_acc:
        best_acc = avg_predict_accuracy
        save_model(model, epoch_i, timestamp)
    
        

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
