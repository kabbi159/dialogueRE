import torch
from torch import nn
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
import wandb
from util import process_long_input


class Baseline(pl.LightningModule):
    def __init__(self, bert, config, args):
        super().__init__()
        self.num_labels = config.num_labels
        self.wandb = args.wandb

        self.start_tokens = [config.cls_token_id]
        if config.transformer_type == "bert":
            self.end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            self.end_tokens = [config.sep_token_id, config.sep_token_id]

        self.bert = bert
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.lr = args.lr
        self.num_steps = 0
        self.correct_gt = 0
        self.correct_sys = 0
        self.all_sys = 0

    def forward(self, x):  # in lightning, forward defines the prediction/inference actions
        outputs = self.bert(x)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

    def training_step(self, batch, batch_idx): # training_step defined the train loop. It is independent of forward
        self.num_steps += 1
        input_ids = batch[0]
        attention_mask = batch[1]
        labels = batch[2]

        # outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # pooled_output = outputs[1] # cls token (pooler output)
        sequence_ouput, attention = process_long_input(self.bert, input_ids, attention_mask, self.start_tokens, self.end_tokens)
        pooled_output = self.dropout(sequence_ouput[:, 0, :])
        logits = self.classifier(pooled_output)

        loss_fct = nn.BCEWithLogitsLoss()  # labels에 대하여 one hot encoding vector가 아닌 indices를 줘야 한다.
        loss = loss_fct(logits.view(-1, self.num_labels), labels.type_as(logits))
        if self.wandb:
            wandb.log({"loss": loss.item()}, step=self.num_steps)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch[0]
        attention_mask = batch[1]
        labels = batch[2].tolist()

        # outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # pooled_output = outputs[1]  # cls token (pooler output)
        sequence_ouput, attention = process_long_input(self.bert, input_ids, attention_mask, self.start_tokens, self.end_tokens)
        pooled_output = self.dropout(sequence_ouput[:, 0, :])
        logits = self.classifier(pooled_output)
        logits = torch.sigmoid(logits)

        return {'pred': logits, 'label': labels}

    def validation_epoch_end(self, outputs):
        best_p, best_r, best_f = 0., 0., 0.
        best_t = 0.
        preds = []
        labels = []
        for x in outputs:
            labels += x['label']

        for i in range(0, 51):
            for x in outputs:
                logits = x['pred']
                # pred_t = torch.where(logits >= i/100, 1, 0).tolist()
                # print(pred_t)
                pred = get_predict(logits.tolist(), T2=i/100)
                preds += pred

            precision, recall, f_1 = get_evaluate(preds, labels)
            if f_1 > best_f:
                best_f = f_1
                best_t = i
            preds = []

        preds = []
        for x in outputs:
            logits = x['pred']
            # pred_t = torch.where(logits >= best_t/100, 1, 0).tolist()
            pred = get_predict(logits.tolist(), T2=best_t/100)
            preds += pred

        precision, recall, f_1 = get_evaluate(preds, labels)

        print({"dev_p": precision, "dev_r": recall, "dev_f1": f_1, "best_t": best_t})

        if self.wandb:
            wandb.log({"dev_p": precision, "dev_r": recall, "dev_f1": f_1})

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        return optimizer


def get_evaluate(preds, labels):
    tp, fp, fn = 0, 0, 0
    for i in range(len(preds)):
        for j in range(len(preds[i]) - 1):
            if preds[i][j] == 1 and labels[i][j] == 1:
                tp += 1
            elif preds[i][j] == 1 and labels[i][j] == 0:
                fn += 1
            elif preds[i][j] == 0 and labels[i][j] == 1:
                fp += 1

    precision = tp / (tp + fp) if (tp+fp) != 0 else 1
    recall = tp / (tp + fn) if (tp+fn) != 0 else 0
    f_1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    return precision, recall, f_1


def get_predict(result, T1=0.5, T2=0.4):
    # 0.5 이상은 result에 들어가고
    # 만약에 result가 아무것도 없고 모든 결과가 0.4 이하이면 out of class (36)
    # 그것이 아니면 그 중 최대값을 갖는 친구 (0.4초과 0.5 미만을 갖는 최대값의 class)

    for i in range(len(result)):
        r = []
        maxl, maxj = -1, -1
        for j in range(len(result[i])):
            if result[i][j] > T1:
                r += [j]
            if result[i][j] > maxl:
                maxl = result[i][j]
                maxj = j
        if len(r) == 0:
            if maxl <= T2:
                r = [36]
            else:
                r += [maxj]
        result[i] = [1 if i in r else 0 for i in range(36)]

    return result