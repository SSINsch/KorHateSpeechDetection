import pandas as pd
import torch
import numpy as np
from utils import prepare_sequence, prepare_char_sequence, preprocess_label
from torch.utils.data import DataLoader
SIZE_MAX_CNN = 5


class KorHateDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, tok2idx, char2idx):
        # super(DocumentDataset, self).__init__()
        self.data = pd.read_csv(data_path, sep='\t')
        self.comments = self.data['comments']
        self.hate = self.data['hate']
        self.tokenizer = tokenizer
        self.tok2idx = tok2idx
        self.char2idx = char2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x_comment = self.comments[index]
        y_hate_item = self.hate[index]

        # check if there is x_comment == nan case
        # if so, replace it with dot('.')
        if type(x_comment) == float:
            x_comment = '.'

        x_comment_tokenized = self.tokenizer.morphs(x_comment)

        # 문장 내의 토큰마다 한 글자씩 뜯어서 보관
        x_comment_char_item = []
        for x_word in x_comment_tokenized:
            x_char_item = []

            for x_char in x_word:
                x_char_item.append(x_char)

            x_comment_char_item.append(x_char_item)

        # get index of words(token) and characters(of each word)
        x_idx_tokens_item = prepare_sequence(x_comment_tokenized, self.tok2idx)
        x_idx_char_item = prepare_char_sequence(x_comment_char_item, self.char2idx)
        y_hate_item = preprocess_label(y_hate_item)

        return x_comment_tokenized, x_idx_tokens_item, x_comment_char_item, x_idx_char_item, y_hate_item


def collate_with_char(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)

    x_comment_tokenized, x_idx_tokens_item, x_comment_char_item, x_idx_char_item, y_hate_item = zip(*data)

    # label data Torch화
    targets = torch.tensor(y_hate_item)

    # batch 하에서 문장 내 최대 단어 개수 측정
    max_word_len = int(np.amax([len(word_tokens) for word_tokens in x_idx_tokens_item]))

    # batch size 측정
    batch_size = len(x_idx_tokens_item)

    # 문장이 최대 단어 수가 되도록 패딩
    padded_word_tokens_matrix = np.ones((batch_size, max_word_len), dtype=np.int64)
    for i in range(padded_word_tokens_matrix.shape[0]):
        for j in range(padded_word_tokens_matrix.shape[1]):
            try:
                padded_word_tokens_matrix[i, j] = x_idx_tokens_item[i][j]
            except IndexError:
                pass

    # batch 내 존재하는 토큰들에 대해 최대 길이 측정
    max_char_len = int(np.amax([len(char_tokens) for word_tokens in x_idx_char_item for char_tokens in word_tokens]))
    if max_char_len < SIZE_MAX_CNN:  # size of maximum filter of CNN
        max_char_len = SIZE_MAX_CNN

    # 각 토큰이 최대 길이가 되도록 패딩
    padded_char_tokens_matrix = np.ones((batch_size, max_word_len, max_char_len), dtype=np.int64)
    for i in range(padded_char_tokens_matrix.shape[0]):
        for j in range(padded_char_tokens_matrix.shape[1]):
            for k in range(padded_char_tokens_matrix.shape[1]):
                try:
                    padded_char_tokens_matrix[i, j, k] = x_idx_char_item[i][j][k]
                except IndexError:
                    pass

    padded_word_tokens_matrix = torch.from_numpy(padded_word_tokens_matrix)
    padded_char_tokens_matrix = torch.from_numpy(padded_char_tokens_matrix)

    return x_comment_tokenized, padded_word_tokens_matrix, x_comment_char_item, padded_char_tokens_matrix, targets


def get_loader(data_path, tokenizer, tok2idx, char2idx, batch_size=256, shuffle=True):
    dataset = KorHateDataset(data_path=data_path,
                             tokenizer=tokenizer,
                             tok2idx=tok2idx,
                             char2idx=char2idx)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             collate_fn=collate_with_char)

    return data_loader
