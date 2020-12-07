import json


def read_data(data_dir, tokenizer, max_seq_length=1024):
    max_len = 0
    # read dialogre dataset
    with open(data_dir, "r") as reader:
        data = json.load(reader)

    features = []
    hmm = {}
    real_max_len = 0
    for i in range(len(data)):
        for j in range(len(data[i][1])):
            rid = []
            for k in range(36):
                if k + 1 in data[i][1][j]["rid"]:
                    rid += [1]
                else:
                    rid += [0]

            dialogue = '\n'.join(data[i][0])

            dialogue_input_ids = tokenizer.encode(dialogue)
            if real_max_len < len(dialogue_input_ids):
                real_max_len = len(dialogue_input_ids)
            # dialogue_input_ids = dialogue_input_ids[:490]
            cls_input_ids = [tokenizer.cls_token_id]
            sep_input_ids = [tokenizer.sep_token_id]
            x_input_ids = tokenizer.encode(data[i][1][j]["x"])
            y_input_ids = tokenizer.encode(data[i][1][j]["y"])

            input_ids = cls_input_ids + dialogue_input_ids + sep_input_ids + x_input_ids + sep_input_ids + y_input_ids + sep_input_ids

            # print(len(input_ids))
            if len(input_ids) > max_len:
                max_len = len(input_ids)
            feature = {'input_ids': input_ids,
                       'labels': rid
            }

            if sum(rid) in hmm:
                hmm[sum(rid)] += 1
            else:
                hmm[sum(rid)] = 1

            features.append(feature)

    print("max len:", max_len)  # 833
    print(hmm)

    return features

def read_test(data_dir, tokenizer):
    max_len = 0
    # read dialogre dataset
    with open(data_dir, "r") as reader:
        data = json.load(reader)
    features = []
    for i in range(len(data)):
        for j in range(len(data[i][1])):
            dialogue = '\n'.join(data[i][0])

            dialogue_input_ids = tokenizer.encode(dialogue)

            cls_input_ids = [tokenizer.cls_token_id]
            sep_input_ids = [tokenizer.sep_token_id]
            x_input_ids = tokenizer.encode(data[i][1][j]["x"])
            y_input_ids = tokenizer.encode(data[i][1][j]["y"])

            input_ids = cls_input_ids + dialogue_input_ids + sep_input_ids + x_input_ids + sep_input_ids + y_input_ids + sep_input_ids

            # print(len(input_ids))
            if len(input_ids) > max_len:
                max_len = len(input_ids)
            feature = {'input_ids': input_ids,
                       }

            features.append(feature)

    print("max len:", max_len)  # 833

    return features

if __name__ == "__main__":
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    config = AutoConfig.from_pretrained(
        'bert-base-cased',
        num_labels=36
    )
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    read_data('data/test.json', tokenizer)
