import random
import warnings
import pandas as pd

from sklearn.metrics import f1_score, classification_report
import torch
import torch.nn as nn
from konlpy.tag import Mecab

from korean_hate_speech_review_data_loader import get_loader
from models import CNNfeaturedBiLSTM
from utils import to_np, to_var, build_token_vocab, build_char_vocab

warnings.filterwarnings(action='ignore')

if __name__ == "__main__":
    # 각종 전역변수들
    char_emb_dim = 32
    char_vocab_size = 5000
    data_path = 'data\\korean-hate-speech\\labeled'
    path_train_data = 'data\\korean-hate-speech\\labeled\\new_train.tsv'
    path_test_data = 'data\\korean-hate-speech\\labeled\\new_test.tsv'
    BATCH_SIZE = 128
    TEST_BATCH_SIZE = 1000
    HIDDEN_DIM = 128
    NUM_LSTM_LAYER = 2
    n_classes = 3
    learning_rate = 0.001
    tokenizer = Mecab(r'C:\mecab\mecab-ko-dic')
    num_epoch = 10

    # 시드 고정
    SEED = 5
    random.seed(SEED)
    torch.manual_seed(SEED)

    # CUDA setting 확인
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

    # vocab set
    data = pd.read_csv(path_train_data, sep='\t')
    comments = data['comments']
    tok_vocab = build_token_vocab(comments, tokenizer)
    char_vocab = build_char_vocab(comments, tokenizer)

    # model
    cnn_feat_bilstm_model = CNNfeaturedBiLSTM(hidden_dim=HIDDEN_DIM,
                                              num_lstm_layer=NUM_LSTM_LAYER,
                                              n_classes=n_classes,
                                              vocab_size=len(tok_vocab[1]),
                                              char_vocab_size=len(char_vocab[1]))

    cnn_feat_bilstm_model = cnn_feat_bilstm_model.to(DEVICE)
    optimizer = torch.optim.Adam(cnn_feat_bilstm_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # load data
    data_loader = get_loader(data_path=path_train_data,
                             batch_size=BATCH_SIZE,
                             tokenizer=tokenizer,
                             tok2idx=tok_vocab[1],
                             char2idx=char_vocab[1])

    test_loader = get_loader(data_path=path_test_data,
                             batch_size=TEST_BATCH_SIZE,
                             tokenizer=tokenizer,
                             tok2idx=tok_vocab[1],
                             char2idx=char_vocab[1])

    # train
    max_macro_f1_score = 0
    total_step = len(data_loader)
    for epoch in range(num_epoch):
        cnn_feat_bilstm_model.train()
        for step, (x_comment_tokenized, padded_word_tokens_matrix,
                   x_comment_char_item, padded_char_tokens_matrix, y_hate_item) in enumerate(data_loader):
            # gpu 로 보내고
            padded_word_tokens_matrix = to_var(padded_word_tokens_matrix)
            padded_char_tokens_matrix = to_var(padded_char_tokens_matrix)
            y_hate_item = to_var(y_hate_item)

            # 초기화 해주고
            optimizer.zero_grad()

            # 모델에 입력
            pred = cnn_feat_bilstm_model(padded_word_tokens_matrix, padded_char_tokens_matrix)

            # label과 비교해서 loss 구하고 optimizer 처리
            max_predictions, argmax_predictions = pred.max(1)
            loss = criterion(pred, y_hate_item)
            loss.backward()
            optimizer.step()

            # Acc
            accuracy = (y_hate_item == argmax_predictions).float().mean()
            score = f1_score(to_np(y_hate_item), to_np(argmax_predictions), average='macro')
            print(f"[ Training ] >> Epoch [{epoch + 1}/{num_epoch}], Step [{step + 1}/{total_step}]")
            print(f"\t\t Loss: {loss.data:.4f},")
            print(f"\t\t Accuracy: {accuracy.data:.4f}, macro-avg f1: {score:.4f}")

        # Test
        cnn_feat_bilstm_model.eval()

        argmax_labels_list = []
        argmax_predictions_list = []

        for step, (test_x_comment_tokenized, test_padded_word_tokens_matrix,
                   test_x_comment_char_item, test_padded_char_tokens_matrix, test_y_hate_item) in enumerate(test_loader):
            # gpu 로 보내고
            test_padded_word_tokens_matrix = to_var(test_padded_word_tokens_matrix, volatile=True)
            test_padded_char_tokens_matrix = to_var(test_padded_char_tokens_matrix)
            test_y_hate_item = to_var(test_y_hate_item, volatile=True)

            pred = cnn_feat_bilstm_model(test_padded_word_tokens_matrix, test_padded_char_tokens_matrix)

            max_predictions, argmax_predictions = pred.max(1)
            argmax_predictions_list.append(argmax_predictions)
            argmax_labels_list.append(test_y_hate_item)

        # Acc
        argmax_labels = torch.cat(argmax_labels_list, 0)
        argmax_predictions = torch.cat(argmax_predictions_list, 0)
        accuracy = (argmax_labels == argmax_predictions).float().mean()

        # f1 score
        argmax_labels_np_array = to_np(argmax_labels)
        argmax_predictions_np_array = to_np(argmax_predictions)
        macro_f1_score = f1_score(argmax_labels_np_array, argmax_predictions_np_array, average='macro')
        if max_macro_f1_score < macro_f1_score:
            max_macro_f1_score = macro_f1_score

        print(f"[ Test ] >> Epoch [{epoch + 1}/{num_epoch}]")
        print(f"\t\t Accuracy: {accuracy.data:.4f}, macro-avg f1: {score:.4f}")

        print("[ classification_report ]")
        target_names = ['none', 'offensive', 'hate']
        print(classification_report(argmax_labels.cpu().data.numpy(), argmax_predictions.cpu().data.numpy(),
                                    target_names=target_names))

        cnn_feat_bilstm_model.train()
