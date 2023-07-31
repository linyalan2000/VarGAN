import argparse
import os
import random
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import pytorch_lightning as pl
import wandb
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoModel
import sys
import math
sys.path.append('/home/lyl/VarCLR')
from varclr.data.dataset import RenamesDataModule
from varclr.models.model import Model
from varclr.models.tokenizers import PretrainedTokenizer
from varclr.utils.options import add_options
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ["WANDB_API_KEY"] = '80237610117d9e7c23b9f82edf4a363b9fc18f88'  # 将引号内的+替换成自己在wandb上的一串值
    os.environ["WANDB_MODE"] = "offline"
    dm = RenamesDataModule(
        args.train_data_file, args.valid_data_file, args.test_data_files, args
    )
    if not os.path.exists(args.vocab_path):
        dm.setup()

    model = Model(args)
    if args.load_file is not None:
        model = model.load_from_checkpoint(args.load_file, args=args, strict=False)
    model.datamodule = dm

    if not args.test and "bert" in args.sp_model and args.model != "bert":
        # Load pre-trained word embeddings from bert
        bert = AutoModel.from_pretrained(args.sp_model)
        for word, idx in torch.load(args.vocab_path).items():
            try:
                model.encoder.embedding.weight.data[
                    idx
                ] = bert.embeddings.word_embeddings.weight.data[int(word)]
            except ValueError:
                pass
        del bert
    if "bert" in args.model:
        PretrainedTokenizer.set_instance(args.bert_model)

    if args.valid_data_file is not None:
        callbacks = [
            EarlyStopping(
                monitor=f"spearmanr/val_{os.path.basename(dm.valid_data_file)}",
                mode="max",
                patience=args.patience,
            ),
            ModelCheckpoint(
                monitor=f"spearmanr/val_{os.path.basename(dm.valid_data_file)}",
                mode="max"
            ),
        ]
    else:
        callbacks = [
            EarlyStopping(
                monitor=f"loss/val_{os.path.basename(dm.train_data_file)}",
                patience=args.patience,
            ),
            ModelCheckpoint(monitor=f"loss/val_{os.path.basename(dm.train_data_file)}"
                            ),
        ]

    wandb_logger = WandbLogger(name=args.name, project="varclr", log_model=True)
    wandb_logger.log_hyperparams(args)
    args = argparse.Namespace(**wandb_logger.experiment.config)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=wandb_logger,
        gpus=args.gpu,
        auto_select_gpus=args.gpu > 0,
        gradient_clip_val=args.grad_clip,
        callbacks=callbacks,
        progress_bar_refresh_rate=10,
        val_check_interval=0.25,
        limit_train_batches=args.limit_train_batches,
    )

    if not args.test:
        trainer.fit(model, datamodule=dm)
        # will automatically load and test the best checkpoint instead of the last model
        trainer.test(datamodule=dm)
    else:
        # save in hf transformer ckpt format
        trainer.test(model, datamodule=dm)
