'''
Fine-tuning the  with VarGAN in summary tasks 
'''
import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
import numpy as np
import time
import datetime
import random
import os
from utils.data_loader import BertData
from utils.discremiter import GenerationModel, PredictHead, RobertaClassificationHead
from torch import nn
torch.cuda.set_device(3) 
tokenizer = RobertaTokenizer.from_pretrained('microsoft/graphcodebert-base')
encoder = RobertaModel.from_pretrained('microsoft/graphcodebert-base')
config = RobertaConfig.from_pretrained('microsoft/graphcodebert-base')
best_loss=1e9
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config.num_layers = 1
config.output_size = 50265
head  = PredictHead(config)

classifier = RobertaClassificationHead(config)
model = GenerationModel(encoder, head, classifier, config, tokenizer)

model.train().to(device)
optimizer_gen = torch.optim.AdamW(encoder.parameters(),
                  lr = 1e-6, # args.learning_rate - default is 5e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8
                )
optimizer_head = torch.optim.AdamW(head.parameters(),
                  lr = 1e-6, # args.learning_rate - default is 5e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8
                )
optimizer_dis = torch.optim.AdamW(classifier.parameters(),
                  lr = 5e-5, # args.learning_rate - default is 5e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8
                )
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))
seed_val = 114
def save_model(model, epoch, timestamp, name):
    """Save model parameters to checkpoint"""
    os.makedirs(f'./save_model', exist_ok=True)
    ckpt_path=f'./save_model/vargan_{epoch}.pkl'
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
epochs = 10



# ensure a certain output when running the code
###############################################################################
# Load data
###############################################################################
train_set=BertData('data/java_train_data.jsonl')
valid_set=BertData('data/java_valid_data.jsonl')
train_loader=torch.utils.data.DataLoader(dataset=train_set, batch_size=32, shuffle=True, num_workers=1)
valid_loader=torch.utils.data.DataLoader(dataset=valid_set, batch_size=32, shuffle=False, num_workers=1)
print("Loaded data!")

total_t0 = time.time()
best_acc = 0
not_increase_num = 0
for epoch_i in range(0, epochs):
    if not_increase_num == 2: 
        break
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
    total_dis_loss = 0
    for step, batch in enumerate(train_loader):
        # output the progress information
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}. Loss = {:} Dis_loss = {:} Elapsed: {:}.'.format(step, len(train_loader), total_train_loss / 40, total_dis_loss / 40, elapsed))
            total_train_loss = 0
            total_dis_loss = 0
        
        
        # data
        code_input = batch[0].to(device)
        code_attn = code_input.ne(1)
        label = batch[1].to(device)
        code_label = batch[2].to(device)
        # reset grad
        model.zero_grad()  
        loss,_,__ = model(code_input, label, code_label)
        (gen_loss, dis_loss) = loss
        # total loss
        total_train_loss += gen_loss.item()
        total_dis_loss += dis_loss.item()
        # backward
        if step % 2 == 0:
            optimizer_gen.zero_grad()
            optimizer_head.zero_grad()
            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_gen.step()
            optimizer_head.step()

        optimizer_dis.zero_grad()
        dis_loss.backward()
        optimizer_dis.step()
           
    
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
    total_pred_accuracy = 0
    total_name_ok_acc = 0
    total_eval_loss = 0
    # Evaluate data for one epoch
    cnt = 0
    for batch in valid_loader:
        cnt += 1
        # data
        code_input = batch[0].to(device)
        code_attn = code_input.ne(1)
        label = batch[1].to(device)
        code_label = batch[2].to(device)
        # reset grad 
        with torch.no_grad():    
            loss, idx, classifier_output = model(code_input, label, code_label)
        acc_cnt = 0
        total_cnt = 0
        name_ok_cnt = 0
        avg_org_simi = 0
        avg_pred_simi = 0
        batch_acc = 0
        mask_cnt = 0
        # mlm accuracy
        _, pred_ids = torch.max(idx, dim = 1)
        for k in range(len(pred_ids)):
            acc = 0
            for i in range(len(pred_ids[k])):
                if code_input[k][i] == 50264:
                    mask_cnt += 1
                    if pred_ids[k][i] == code_label[k][i]:
                        acc += 1
                
            if mask_cnt != 0:
                acc /= mask_cnt 
            else:
                acc = 0
            batch_acc += acc

        _, pred = torch.max(classifier_output, dim = 1)
        for batch_idx in range(0, len(code_input)):
            if pred[batch_idx] == label[batch_idx]:
                acc_cnt += 1
            total_cnt += 1
        # total loss
        total_eval_loss += loss[0].item()
        total_eval_accuracy += acc_cnt / total_cnt
        total_pred_accuracy += batch_acc / len(batch)
    # print accuracy for this epoch
    avg_val_accuracy = total_eval_accuracy / len(valid_loader)
    avg_pred_accuracy = total_pred_accuracy / len(valid_loader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    print("  Pred Accuracy: {0:.2f}".format(avg_pred_accuracy))
    # loss for this epoch
    avg_val_loss = total_eval_loss / len(valid_loader)
    validation_time = format_time(time.time() - t0)
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))
    #save model
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        save_model(model, epoch_i, timestamp, name=None)    # loss for this epoch
    else:
        not_increase_num += 1
    
    
        

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
