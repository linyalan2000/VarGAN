import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch import optim
from varclr.models.encoders import Encoder, RobertaClassificationHead
from varclr.models.loss import NCESoftmaxLoss
from torch.nn import CrossEntropyLoss, MSELoss

class Model(pl.LightningModule):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.loss_nec = NCESoftmaxLoss(args.nce_t)
        self.loss_cross = CrossEntropyLoss()
        args.vocab_size = len(torch.load(args.vocab_path))
        args.parentmodel = self
        self.encoder = Encoder.build(args)
        self.classifier = RobertaClassificationHead()

    def _forward(self, batch):
        (x_idxs, x_lengths), (y_idxs, y_lengths), x_label, y_label = batch
        # print(x_idxs, x_lengths)
        x_ret = self.encoder(x_idxs, x_lengths)
        y_ret = self.encoder(y_idxs, y_lengths)

        x_pred = self.classifier(x_ret[0])
        y_pred = self.classifier(y_ret[0])
        for i in range(len(x_label)):
            x_label[i] = 1 - x_label[i]
            y_label[i] = 1 - y_label[i]

        # loss = loss_fct(logits, labels)
        # 这里如果是生成器要反一下的
        if self.args.label == 'new':
            return self.loss_nec(x_ret, y_ret) + self.loss_cross(x_pred, x_label) +  self.loss_cross(y_pred, y_label)
        else:
            return self.loss_nec(x_ret, y_ret)


    def classifier_forward(self, batch):
        (x_idxs, x_lengths), (y_idxs, y_lengths), x_label, y_label = batch
        x_ret = self.encoder(x_idxs, x_lengths)
        y_ret = self.encoder(y_idxs, y_lengths)
        x_pred = self.classifier(x_ret[0].detach())
        y_pred = self.classifier(y_ret[0].detach())
        return self.loss_cross(x_pred, x_label) + self.loss_cross(y_pred, y_label)
        
    def _score(self, batch):
        (x_idxs, x_lengths), (y_idxs, y_lengths) = batch
        x_pooled, _ = self.encoder(x_idxs, x_lengths)
        y_pooled, _ = self.encoder(y_idxs, y_lengths)
        return F.cosine_similarity(x_pooled, y_pooled)

    def training_step(self, batch, batch_idx, optimizer_idx):

        # train generator
        if optimizer_idx == 0:
            loss = self._forward(batch).mean()
            self.log("loss/train", loss)
            return loss
 
        # train discriminator
        if optimizer_idx == 1:
            loss = self.classifier_forward(batch).mean()
            self.log("dis_loss/train", loss)
            return loss

    def _unlabeled_eval_step(self, batch, batch_idx):
        loss = self._forward(batch)
        return dict(loss=loss.detach().cpu())

    def _labeled_eval_step(self, batch, batch_idx):
        *batch, labels = batch
        scores = self._score(batch)
        return dict(scores=scores.detach().cpu(), labels=labels.detach().cpu())

    def _shared_eval_step(self, batch, batch_idx):
        if len(batch) == 3:
            return self._labeled_eval_step(batch, batch_idx)
        else:
            return self._unlabeled_eval_step(batch, batch_idx)

    def _unlabeled_epoch_end(self, outputs, prefix):
        # loss = torch.tensor([o["loss"] for o in outputs]).mean()
        loss = torch.cat([o["loss"] for o in outputs]).mean()
        self.log(f"loss/{prefix}", loss)

    def _labeled_epoch_end(self, outputs, prefix):
        scores = torch.cat([o["scores"] for o in outputs]).tolist()
        labels = torch.cat([o["labels"] for o in outputs]).tolist()
        self.log(f"pearsonr/{prefix}", pearsonr(scores, labels)[0])
        self.log(f"spearmanr/{prefix}", spearmanr(scores, labels).correlation)
        filename = 'result.txt'
        # 如果文件存在，则读取文件中的值并加上score，否则创建文件并写入score
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                # 读取文件中的值并转换为浮点数
                value = float(f.read())
                # 加score
                if self.args.label == 'new':
                    value += spearmanr(scores, labels).correlation
                elif self.args.label == 'org':
                    value -= spearmanr(scores, labels).correlation
            with open(filename, 'w') as f:
                # 将新值写入文件
                f.write(str(value))
        else:
            with open(filename, 'w') as f:
                # 如果文件不存在，则写入score
                f.write(str(spearmanr(scores, labels).correlation))

    def _shared_epoch_end(self, outputs, prefix):
        if "labels" in outputs[0]:
            self._labeled_epoch_end(outputs, prefix)
        else:
            self._unlabeled_epoch_end(outputs, prefix)

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_eval_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        self._shared_epoch_end(
            outputs,
            f"val_{os.path.basename(self.datamodule.val_dataloader().dataset.data_file)}",
        )

    def test_epoch_end(self, outputs):
        if isinstance(outputs[0], list):
            for idx, subset_outputs in enumerate(outputs):
                self._shared_epoch_end(
                    subset_outputs,
                    f"test_{os.path.basename(self.datamodule.test_dataloader()[idx].dataset.data_file)}",
                )
        else:
            self._shared_epoch_end(
                outputs,
                f"test_{os.path.basename(self.datamodule.test_dataloader().dataset.data_file)}",
            )

    def configure_optimizers(self):

        # optimizer = optim.Adam([{'encoder_params': self.encoder.parameters()}, {'decoder_params': self.decoder.parameters()}], lr=learning_rate, weight_decay=weight_decay)

        opt_g = optim.Adam(self.encoder.parameters(), lr=self.args.lr)
        opt_d = optim.Adam(self.classifier.parameters(), lr=self.args.dis_lr)
        # return {"bert": optim.AdamW}.get(self.args.model, optim.Adam)(
        #     self.parameters(), lr=self.args.lr
        # )

        return [opt_g, opt_d], []