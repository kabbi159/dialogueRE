import argparse
import wandb

import torch
from torch.utils.data import DataLoader

from transformers import AutoConfig, AutoModel, AutoTokenizer
import pytorch_lightning as pl

from model import Baseline
from prepro import read_data
from util import set_seed, collate_fn


def main():
    # argument parsing
    parser = argparse.ArgumentParser()

    parser.add_argument("--project_name", default='dialogueRE-jiwung', type=str)
    parser.add_argument("--train_dir", default='./data/train.json', type=str)
    parser.add_argument("--val_dir", default='./data/dev.json', type=str)
    parser.add_argument("--test_dir", default='./data/test.json', type=str)
    parser.add_argument("--wandb", default=False, type=str)

    parser.add_argument("--train_batch_size", default=2, type=int)  # bert: 8, roberta: 4
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name", default='bert-base-cased', type=str)
    parser.add_argument("--num_labels", default=36, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)

    args = parser.parse_args()
    if args.wandb:
        wandb.init(project=args.project_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # config, tokenizer, model
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=args.num_labels
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    #print(tokenizer)

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    bert = AutoModel.from_pretrained(
        args.model_name,
        config=config
    )

    set_seed(args)

    # load data

    train_features = read_data(args.train_dir, tokenizer)
    train_dataloader = DataLoader(train_features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8)

    val_features = read_data(args.val_dir, tokenizer)
    val_dataloader = DataLoader(val_features, batch_size=args.train_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8)


    # my model
    model = Baseline(bert, config, args)

    trainer = pl.Trainer(gpus=1, max_epochs=20, num_sanity_val_steps=0, logger=False)  # num_santiny
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    main()

