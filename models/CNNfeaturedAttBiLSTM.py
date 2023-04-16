import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


warnings.filterwarnings(action='ignore')


class CNNfeaturedAttBiLSTM(nn.Module):
    def __init__(self, hidden_dim, num_lstm_layer, n_classes, pre_embedding=None,
                 vocab_size=10000, embed_size=200, char_vocab_size=5000, char_embed_size=50, dropout=0.3):
        super(CNNfeaturedAttBiLSTM, self).__init__()

        # variable for cnn
        channel_input_word = 1
        kernel_sizes = [3]
        # kernel_sizes = [3, 4, 5]
        # LSTM도 -> nn.LSTM(input_size=self.embedding.embedding_dim + (self.char_emb_dim)*len(kernel_sizes),
        channel_output = char_embed_size
        self.char_emb_dim = char_embed_size
        self.char_vocab_size = char_vocab_size

        # variable for bilstm
        self.hidden_dim = hidden_dim
        self.num_lstm_layer = num_lstm_layer
        self.n_classes = n_classes

        # char_cnn layer 세팅해야하지만 사이즈가 동적이기 때문에(배치마다 최대길이가 달라서)
        # forward에서 한다.
        self.char_emb = nn.Embedding(self.char_vocab_size,
                                     self.char_emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv2d(channel_input_word, channel_output, kernel_size=(kernel_size, self.char_emb_dim)) for kernel_size in kernel_sizes])

        # word embedding layer 세팅
        if pre_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pre_embedding, freeze=False)
        else:
            self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=embed_size)
        # BiLSTM layer 세팅
        self.bi_lstm = nn.LSTM(input_size=self.embedding.embedding_dim + self.char_emb_dim,
                               hidden_size=self.hidden_dim,
                               num_layers=self.num_lstm_layer,
                               dropout=dropout,
                               batch_first=True,
                               bidirectional=True)

        # last linear layer 세팅
        # bidirectional 이라서 hidden_dim * 2
        self.linear = nn.Linear(self.hidden_dim * 2, self.n_classes)

    def attention_net(self, lstm_output, final_hidden_state):
        """
        attention network
        :param lstm_output: [batch, len_seq, hidden_dim * num_directions(=2)]
        :param final_hidden_state: [num_directions, batch, hidden_dim]
        :return:
        """
        # hidden : [batch, hidden_dim * num_directions(=2), n_layer(=1))]
        # only use last layer of lstm
        hidden = final_hidden_state.view(-1, self.hidden_dim * 2, 1)

        # bmm = batch matrix multiplication = [batch, n, m] * [batch, m, p] = [B, n, p]
        # attn_weights : [batch, len_seq]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)

        # context : [batch, hidden_dim * num_directions(=2)]
        # = [batch, hidden_dim * num_directions(=2), len_seq] * [batch, len_seq, 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2))
        context = context.squeeze(2)

        return context, soft_attn_weights

    def forward(self, sentence, characters):
        # embedding
        # word_embedded:  [batch_size, word_num_per_sentence, word_embedding_size]
        word_embedded = self.embedding(sentence)
        # print('word_embedded: ', word_embedded.shape)

        char_output = []
        for i in range(characters.size(1)):
            # char_embedded = (batch, channel_input, words, word_embedding)
            char_embedded = self.char_emb(characters[:, 1]).unsqueeze(1)

            h_convs = [nn.functional.relu(conv(char_embedded)).squeeze(3) for conv in self.convs]

            # [(batch,channel_out), ...]*len(kernel_sizes)
            h_pools = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in h_convs]

            # 리스트 쌓아서 하나로 합치기
            h_pools = torch.cat(h_pools, 1)

            # 단어단위임을 고려해서 (batch, 1, Maxpooled_dim)으로 다시 바꿔줌
            out = h_pools.unsqueeze(1)
            char_output.append(out)

        # 이제 단어 단어끼리 붙여줌
        char_output = torch.cat(char_output, 1)

        # shape = (batch, seq_Length, hidden_dim of input)
        embedded = torch.cat([word_embedded, char_output], dim=-1)

        # bilstm
        # lstm_out = (Batch, Length, hidden_dim * (1 or 2))
        # h_n = (num_layer * (1 or 2), batch, hidden_dim)
        lstm_out, (h_n, c_n) = self.bi_lstm(embedded)

        # forward와 backward의 마지막 time-step의 은닉 상태를 가지고 와서 concat
        # 이때 모델이 batch_first라는 점에 주의한다.
        # attention 도입으로 concat 말고 backward와 forward의 마지막 층만 사용
        # hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        hidden = h_n[-2:, :, :]

        # attention network
        attn_out, attention_value = self.attention_net(lstm_out, hidden)

        # linear layer
        out = self.linear(attn_out)

        return out, attention_value
