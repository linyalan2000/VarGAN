import torch
from torch import nn 
import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, RobertaForCausalLM, RobertaForSequenceClassification

import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        # self.lstm = nn.LSTM(config., config.output_size, config.num_layers, batch_first=True)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)


    def forward(self, features, **kwargs):

        x = torch.mean(features, dim = 1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
class RobertaTokenClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)


    def forward(self, features, **kwargs):
        features = self.dropout(features)
        logits = self.dense(features)
        logits = torch.tanh(logits)
        logits = self.dropout(logits)
        logits = self.out_proj(logits)
        return logits


class PredictHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.decoder = nn.Linear(config.hidden_size, config.output_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x


class Model(nn.Module):   
    def __init__(self, encoder, config, tokenizer, encoder_path = None, classifier_dict = None):
        super(Model, self).__init__()
        self.encoder = encoder
        if encoder_path != None:
            self.encoder.load_state_dict(torch.load(encoder_path))
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        if classifier_dict != None:
            self.classifier.load_state_dict(classifier_dict)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=8)

    def forward(self, input_ids=None,labels=None): 
        input_ids=input_ids.view(-1,256)
        outputs1 = self.encoder(input_ids = input_ids, attention_mask=input_ids.ne(1))[0].detach()
        # another choise: a transformer
        # outputs2 = self.transformer_encoder(outputs1)
        logits=self.classifier(outputs1)
        prob=F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            # loss2 = loss_fct(logits., labels)
            return loss, prob
        else:
            return prob
        
class GenerationModel(nn.Module):   
    def __init__(self, encoder, head, classifier, config, tokenizer, encoder_path = None, decoder_path = None, classifier_path = None):
        super().__init__()
        self.encoder = encoder
        if encoder_path != None:
            self.encoder.load_state_dict(torch.load(encoder_path))
        self.config=config
        self.tokenizer=tokenizer
        self.decoder=head
        if decoder_path != None:
            self.decoder.load_state_dict(torch.load(decoder_path))
        self.classifier=classifier

    def forward(self, input_ids=None,labels=None, code_labels=None): 
        input_ids=input_ids.view(-1,256)
        outputs = self.encoder(input_ids = input_ids, attention_mask=input_ids.ne(1))[0]
        # another choise: a transformer
        # outputs2 = self.transformer_encoder(outputs1)
        logits=self.decoder(outputs).permute(0,2,1)
        
        classifer_output = self.classifier(outputs)
        disloss_output = self.classifier(outputs.detach())
        prob=F.softmax(logits, dim=1)
        fake_labels = torch.zeros(len(labels)).long().to('cuda')
        if code_labels is not None and labels is not None:
            loss_fct = CrossEntropyLoss()
            loss_mlm = loss_fct(logits, code_labels)
            for i in range(len(labels)):
                fake_labels[i] = 1 - int(labels[i])
            loss_label = loss_fct(classifer_output, fake_labels)
            loss = loss_label + loss_mlm
            loss_dis = loss_fct(disloss_output, labels)
            return (loss, loss_dis), prob, classifer_output
        else:
            return prob
        
class GenerationGanOnylModel(nn.Module):   
    def __init__(self, encoder, head, classifier, config, tokenizer, encoder_path = None, decoder_path = None, classifier_path = None):
        super().__init__()
        self.encoder = encoder
        if encoder_path != None:
            self.encoder.load_state_dict(torch.load(encoder_path))
        self.config=config
        self.tokenizer=tokenizer
        self.decoder=head
        if decoder_path != None:
            self.decoder.load_state_dict(torch.load(decoder_path))
        self.classifier=classifier

    def forward(self, input_ids=None,labels=None, code_labels=None): 
        input_ids=input_ids.view(-1,256)
        outputs = self.encoder(input_ids = input_ids, attention_mask=input_ids.ne(1))[0]
        # another choise: a transformer
        # outputs2 = self.transformer_encoder(outputs1)
        logits=self.decoder(outputs).permute(0,2,1)
        
        classifer_output = self.classifier(outputs)
        disloss_output = self.classifier(outputs.detach())
        prob=F.softmax(logits, dim=1)
        fake_labels = torch.zeros(len(labels)).long().to('cuda')
        if code_labels is not None and labels is not None:
            loss_fct = CrossEntropyLoss()
            for i in range(len(labels)):
                fake_labels[i] = 1 - int(labels[i])
            loss_label = loss_fct(classifer_output, fake_labels)
            loss = loss_label
            loss_dis = loss_fct(disloss_output, labels)
            return (loss, loss_dis), prob, classifer_output
        else:
            return prob
        
class GenerationMIPOnylModel(nn.Module):   
    def __init__(self, encoder, head, classifier, config, tokenizer, encoder_path = None, decoder_path = None, classifier_path = None):
        super().__init__()
        self.encoder = encoder
        if encoder_path != None:
            self.encoder.load_state_dict(torch.load(encoder_path))
        self.config=config
        self.tokenizer=tokenizer
        self.decoder=head
        if decoder_path != None:
            self.decoder.load_state_dict(torch.load(decoder_path))

    def forward(self, input_ids=None,labels=None, code_labels=None): 
        input_ids=input_ids.view(-1,256)
        outputs = self.encoder(input_ids = input_ids, attention_mask=input_ids.ne(1))[0]
        # another choise: a transformer
        # outputs2 = self.transformer_encoder(outputs1)
        logits=self.decoder(outputs).permute(0,2,1)
        
        # classifer_output = self.classifier(outputs)
        # disloss_output = self.classifier(outputs.detach())
        prob=F.softmax(logits, dim=1)
        # fake_labels = torch.zeros(len(labels)).long().to('cuda')
        if code_labels is not None and labels is not None:
            loss_fct = CrossEntropyLoss()
            loss_mlm = loss_fct(logits, code_labels)
            # for i in range(len(labels)):
            #     fake_labels[i] = 1 - int(labels[i])
            # loss_label = loss_fct(classifer_output, fake_labels)
            loss = loss_mlm
            # loss_dis = loss_fct(disloss_output, labels)
            return (loss, 0), prob, None
        else:
            return prob

# class GenerationTokenModel(nn.Module):   
#     def __init__(self, encoder, head, classifier_token, classifier_sent, config, tokenizer, encoder_path = None, decoder_path = None, classifier_path = None):
#         super().__init__()
#         self.encoder = encoder
#         if encoder_path != None:
#             self.encoder.load_state_dict(torch.load(encoder_path))
#         self.config=config
#         self.tokenizer=tokenizer
#         self.decoder=head
#         if decoder_path != None:
#             self.decoder.load_state_dict(torch.load(decoder_path))
#         self.classifier_token=classifier_token
#         self.classifier_sent=classifier_sent

#     def forward(self, input_ids=None,labels=None, code_labels=None, token_labels=None, return_cls_result = False): 
#         input_ids=input_ids.view(-1,256)
#         outputs = self.encoder(input_ids = input_ids, attention_mask=input_ids.ne(1))[0]
#         # another choise: a transformer
#         # outputs2 = self.transformer_encoder(outputs1)
#         logits=self.decoder(outputs).permute(0,2,1)
#         classifer_sent_output = self.classifier_sent(outputs)
#         disloss_sent_output = self.classifier_sent(outputs.detach())
#         prob=F.softmax(logits, dim=1)
#         fake_labels = torch.zeros(len(labels)).long().to('cuda')
#         loss_fct = CrossEntropyLoss()
#         loss_mlm = loss_fct(logits, code_labels)
#         for i in range(len(labels)):
#             fake_labels[i] = 1 - int(labels[i])
        
#         loss_label = loss_fct(classifer_sent_output, fake_labels)

#         loss_token = 0
#         loss_dis_token = 0
#         token_classifier_acc = 0
#         total_token_classifier_acc = 0
#         for i in range(len(outputs)):
#             token_classifier_acc = 0
#             classifer_token_output, new_labels, new_fake_labels = self.classifier_token(outputs[i], token_labels[i])
#             disloss_token_output, _, __ = self.classifier_token(outputs[i].detach(), token_labels[i])
#             if len(classifer_token_output):
#                 loss_token += loss_fct(classifer_token_output, torch.tensor(new_fake_labels).to('cuda'))
#                 loss_dis_token += loss_fct(disloss_token_output, torch.tensor(new_labels).to('cuda'))
#             if return_cls_result and len(classifer_token_output):
#                 classifier_token_result = classifer_token_output.argmax(1)
#                 for i in range(len(classifier_token_result)):
#                     if classifier_token_result[i] == new_labels[i]:
#                         token_classifier_acc += 1
#                 if len(new_labels) != 0:
#                     total_token_classifier_acc +=  token_classifier_acc / len(new_labels)
#         loss_token /= len(outputs)
#         loss_dis_token /= len(outputs)
#         total_token_classifier_acc /= len(outputs)
#         loss = loss_label + loss_mlm
#         loss_dis_sent = loss_fct(disloss_sent_output, labels)
#         return (loss, loss_dis_token, loss_dis_sent), prob, classifer_sent_output, total_token_classifier_acc

class GenerationTokenModel(nn.Module):   
    def __init__(self, encoder, head, classifier_token, classifier_sent, config, tokenizer, encoder_path = None, decoder_path = None, classifier_path = None):
        super().__init__()
        self.encoder = encoder
        if encoder_path != None:
            self.encoder.load_state_dict(torch.load(encoder_path))
        self.config=config
        self.tokenizer=tokenizer
        self.decoder=head
        if decoder_path != None:
            self.decoder.load_state_dict(torch.load(decoder_path))
        self.classifier_token=classifier_token
        self.classifier_sent=classifier_sent

    def forward(self, input_ids=None,labels=None, code_labels=None, token_labels=None, fake_token_labels = None, return_cls_result = False): 
        input_ids=input_ids.view(-1,256)
        outputs = self.encoder(input_ids = input_ids, attention_mask=input_ids.ne(1))[0]
        logits=self.decoder(outputs).permute(0,2,1)
        classifer_sent_output = self.classifier_sent(outputs)
        disloss_sent_output = self.classifier_sent(outputs.detach())
        classifer_token_output = self.classifier_token(outputs)
        classifer_token_output = classifer_token_output.permute(0,2,1)
        disloss_token_output = self.classifier_token(outputs.detach()) # [batch_size, length, 2]
        disloss_token_output = disloss_token_output.permute(0,2,1)
        
        # loss for generater 
        prob=F.softmax(logits, dim=1)
        loss_fct = CrossEntropyLoss()
        loss_mlm = loss_fct(logits, code_labels)

        fake_labels = torch.zeros(len(labels)).long().to('cuda')
        for i in range(len(labels)):
            fake_labels[i] = 1 - int(labels[i])
        loss_label = loss_fct(classifer_sent_output, fake_labels)
        loss_token = loss_fct(classifer_token_output, fake_token_labels)
        loss = loss_label + loss_mlm + loss_token

        # loss for sentenc classifier
        loss_dis_sent = loss_fct(disloss_sent_output, labels)

        # loss for token classifier
        loss_dis_token = loss_fct(disloss_token_output, token_labels)

        # return loss and predict result
        return (loss, loss_dis_token, loss_dis_sent), prob, classifer_sent_output, classifer_token_output
