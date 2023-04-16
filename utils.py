import os
import collections
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False


def prepare_sequence(seq, word_to_idx):
    idxs = list()

    for word in seq:
        if word not in word_to_idx.keys():
            idxs.append(word_to_idx['<unk>'])
        else:
            idxs.append(word_to_idx[word])

    return idxs


def prepare_char_sequence(seq, char_to_idx):
    char_idxs = list()

    for word in seq:
        idxs = list()
        for char in word:
            if char not in char_to_idx:
                idxs.append(char_to_idx['<unk>'])
            else:
                idxs.append(char_to_idx[char])
        char_idxs.append(idxs)

    return char_idxs


def preprocess_label(label, only_hate=False):
    if only_hate is True:
        label_matching = {'none': 0, 'offensive': 1, 'hate': 1}
    else:
        label_matching = {'none': 0, 'offensive': 1, 'hate': 2}

    return label_matching[label]


def to_np(x):
    return x.data.cpu().numpy()


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def build_token_vocab(comments, tokenizer, min_freq=3, max_vocab=30000):
    vocab = []
    for i, sentence in enumerate(comments):
        try:
            temp = tokenizer.morphs(sentence)
            vocab.extend(temp)
        except Exception as e:
            # sometimes there is no comment in this dataset
            pass

    vocab = collections.Counter(vocab)
    temp = {}
    for key in vocab.keys():
        if vocab[key] >= min_freq:
            temp[key] = vocab[key]
    vocab = temp
    # vocab = sorted(vocab, key=lambda x: -vocab[x])
    if len(vocab) > max_vocab:
        vocab = vocab[:max_vocab]

    tok2idx = {'<unk>': 0, '<pad>': 1}
    for tok in vocab:
        tok2idx[tok] = len(tok2idx)

    tok_vocab = (vocab, tok2idx)

    return tok_vocab


def build_char_vocab(comments, tokenizer):
    vocab = []
    for sentence in comments:
        try:
            tokens = tokenizer.morphs(sentence)
            for tok in tokens:
                for char in tok:
                    vocab.append(char)
        except Exception as e:
            # sometimes there is no comment in this dataset
            pass

    vocab = list(set(vocab))

    char2idx = {'<unk>': 0, '<pad>': 1}
    for char in vocab:
        char2idx[char] = len(char2idx)

    char_vocab = (vocab, char2idx)

    return char_vocab


def split_save_tsv(path_tsv, path_output, ratio=0.8):
    path_output = os.path.abspath(path_output)
    df = pd.read_csv(path_tsv, sep='\t')
    df = df.sample(frac=1).reset_index(drop=True)

    df_train = df[:int(len(df) * ratio)]
    df_test = df[int(len(df) * ratio):]

    df_train.to_csv(path_output+'\\new_train.tsv', sep='\t', index=False)
    df_test.to_csv(path_output+'\\new_test.tsv', sep='\t', index=False)


def debug_result(pred, argmax_pred, argmax_answer, sent, debug_output, heatmap_name):
    """
    answer pred prob_none prob_offen prob_hate sent 순서
    :param pred:
    :param argmax_pred:
    :param argmax_answer:
    :param sent: sentence
    :param debug_output: debug output file
    :return: none
    """
    pred = np.array(pred).T
    complete_sent = []
    for s in sent:
        complete_sent.append(''.join(s))
    col = ['answer', 'pred', 'prob_none', 'prob_offe', 'prob_hate', 'sentence']
    d = {'answer': replace_num(argmax_answer), 'pred': replace_num(argmax_pred), 'prob_none': pred[0],
         'prob_offe': pred[1], 'prob_hate': pred[2], 'sentence': complete_sent}
    df = pd.DataFrame(data=d, columns=col)
    df_wrong = df[df['answer'] != df['pred']]
    df.to_csv(debug_output, sep='\t')
    df_wrong.to_csv(debug_output.replace('.tsv', '_wrong.tsv'), sep='\t')

    plt.figure()
    cf = confusion_matrix(argmax_answer, argmax_pred)
    sns.heatmap(cf, annot=True, cmap='Blues')
    plt.savefig(heatmap_name)


def replace_num(l):
    label = {0: 'none', 1: 'offe', 2: 'hate'}
    x = []
    for i in l:
        x.append(label[i])

    return x


def plot_att_val(sents, att_vals, predictions, labels):
    for i in range(len(sents)):
        x = sents[i]
        y = to_np(att_vals[i])
        if len(x) < 71:
            temp = [' ']*(71-(len(x)))
            x.extend(temp)
            temp = [0] * (71 - (len(x)))
            temp = np.array(temp)
            y = np.concatenate([y, temp])

        df = pd.DataFrame({"att_val": y},
                          index=x)
        plt.figure(figsize=(15, 15))
        sns.heatmap(df, fmt="g", cmap='viridis')
        name = f'./log/{i:04d}_{labels[i]}_{predictions[i]}.png'
        plt.savefig(name)
