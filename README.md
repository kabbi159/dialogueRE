# dialogueRE
Re-implementation of **"Dialogue-Based Relation Extraction"** (ACL 2020) [[paper]](https://arxiv.org/abs/2004.08056)  
Official code and dataset are avilable in [[code]](https://github.com/nlpdata/dialogre) .  
I want to make baseline code trainable in end-to-end for future works in this dataset.  
I only implemented only bert baseline in DialogRE dataset v1 (English).

## Result in devlopment set
|Model|F1|Precision|Recall|
|---|---|---|---|
|bert (my)|58.0|59.2|56.7|
|bert (paper)|60.6|-|-|


## Usage
Before run this code, construct dataset  ```data/{train,dev,test}.json``` with this format.  
You can run this code:
```bash
python main.py
--transformer_type=bert \
--model_name=bert-base-cased \
--seed=42 \
--lr=3e-5 \
--wandb=True
```

## Issue (check == solved)
- [x] apply wandb and pytorch lighting - but it is my first attempt to use these libraries, please excuse my poor code :)
- [ ] roberta model training failure
- [x] long sequence processing implementation
- [ ] long sequence processing harms performance
