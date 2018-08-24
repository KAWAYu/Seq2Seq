#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
from io import open
import random
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

import model
import my_utils as utils

device = torch.device('cpu')


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src')
    parser.add_argument('-train_tgt')
    parser.add_argument('-valid_src')
    parser.add_argument('-valid_tgt')
    parser.add_argument('--vocab_size', '-vs', default=50000, type=int, help='vocabulary size')
    parser.add_argument('--embed_size', '-es', default=256, type=int, help='embedding size')
    parser.add_argument('--hidden_size', '-hs', default=256, type=int, help='hidden layer size')
    parser.add_argument('--epochs', '-e', default=10, type=int, help='the number of epochs')
    parser.add_argument('--batch_size', '-bs', default=200, type=int, help='batch size')
    parser.add_argument('--gpu_id', '-g', default=-1, type=int, help='GPU id you want to use')
    parser.add_argument('--model_prefix', '-mf', default='model', help='prefix of model name')
    parser.add_argument('--reverse', '-r', action='store_true', default=False, help='reverse order of input sequences')

    args = parser.parse_args()
    return args


def train(train_srcs, train_tgts, valid_srcs, valid_tgts, encoder, decoder, s_vocab, t_vocab,
          num_epochs, batch_size, device, reverse=True):
    """
    訓練する関数
    :param train_srcs: 原言語の訓練データ
    :param train_tgts: 目的言語の訓練データ
    :param valid_srcs: 原言語の開発データ
    :param valid_tgts: 目的言語の開発データ
    :param encoder: エンコーダ側のモデル
    :param decoder: デコーダ側のモデル
    :param s_vocab: 原言語側の語彙辞書 (語彙 -> ID)
    :param t_vocab: 目的言語の語彙辞書 (語彙 -> ID)
    :param num_epochs: エポック数
    :param batch_size: バッチ数
    :param device: CPUかGPUの指定
    :return tuple(train_loss, dev_loss): 訓練データでの各エポックでのロス、開発データでの各エポックでのロス
    """

    # 学習率最適化手法
    encoder_optim = torch.optim.Adam(encoder.parameters(), weight_decay=1e-5)
    decoder_optim = torch.optim.Adam(decoder.parameters(), weight_decay=1e-5)

    criterion = torch.nn.CrossEntropyLoss()  # 損失関数

    t_vocab_list = [k for k, _ in sorted(t_vocab.items(), key=lambda x: x[1])]
    train_losses, dev_losses = [], []

    for e in range(1, num_epochs + 1):
        print('Epoch %d start...' % e)
        total_loss = 0
        indexes = [i for i in range(len(train_srcs))]  # 0..len(train_srcs)のインデックスのリストを作成し、
        random.shuffle(indexes)                        # 順番をシャッフル
        k = 0  # 何文処理をしたかを格納するカウンタ変数
        while k < len(indexes):
            encoder_optim.zero_grad(), decoder_optim.zero_grad()  # 勾配をリセット
            batch_loss = 0
            batch_idx = indexes[k: min(k + batch_size, len(indexes))]  # batch_size分のインデックスを取り出し
            batch_t_s, batch_t_t = [], []
            # バッチ処理用の文対をまとめる
            for idx in batch_idx:
                batch_t_s.append(train_srcs[idx])
                batch_t_t.append(train_tgts[idx])

            batch_src_len = [len(s) + 1 for s in batch_t_s]  # 各バッチの長さ（<EOS>のインデックス + 1）を記録
            # TODO: 各バッチの長さを使って<EOS>タグが入力されたときの隠れ層を取得できるようにする
            max_s_len = max(batch_src_len)
            max_t_len = max(len(s) + 1 for s in batch_t_t)
            # バッチの中で一番長い文に合わせてpadding
            for i in range(len(batch_t_s)):
                if reverse:
                    batch_t_s[i] = (batch_t_s[i] + [s_vocab['<EOS>']] * (max_s_len - len(batch_t_s[i])))[::-1]
                else:
                    batch_t_s[i] = (batch_t_s[i] + [s_vocab['<EOS>']] * (max_s_len - len(batch_t_s[i])))
                batch_t_t[i] = batch_t_t[i] + [t_vocab['<EOS>']] * (max_t_len - len(batch_t_t[i]))

            encoder_init_hidden = encoder.init_hidden(len(batch_idx), device)
            xs = torch.tensor(batch_t_s).to(device)
            ehs = encoder(xs, encoder_init_hidden)

            ys = torch.tensor(batch_t_t).to(device)
            dhidden = ehs  # encoder側の最後の隠れ層をdecoderの隠れ層の初期値に
            pred_words = torch.tensor([[t_vocab['<BOS>']] for _ in range(len(batch_idx))]).to(device)
            pred_seq = [[] for _ in range(len(batch_idx))]
            for j in range(max_t_len - 1):
                preds, dhidden = decoder(pred_words, dhidden)
                _, topi = preds.topk(1)
                batch_loss += criterion(preds, ys[:, j])
                pred_words = ys[:, j + 1].view(-1, 1)
                for i in range(len(pred_seq)):
                    pred_seq[i].append(topi[i])
            i = random.randrange(0, len(batch_idx))
            print(' '.join(t_vocab_list[t] for t in batch_t_t[i]))
            print(' '.join(t_vocab_list[t] if t < len(t_vocab_list) else t_vocab_list[0] for t in pred_seq[i]))

            total_loss += batch_loss.item()
            batch_loss.backward()
            decoder_optim.step()
            encoder_optim.step()
            k += len(batch_idx)
            print('\r%d sentences was learned, loss %.4f' % (k, batch_loss.item()), end='')
        print()
        train_losses.append(total_loss)
        dev_loss = dev_evaluate(valid_srcs, valid_tgts, encoder, decoder, s_vocab, t_vocab, device, reverse=reverse)
        dev_losses.append(dev_loss)
        print('train loss: %.6f, valid loss: %.6f' % (total_loss, dev_loss))
    return train_losses, dev_losses


def dev_evaluate(valid_srcs, valid_tgts, encoder, decoder, s_vocab, t_vocab, device, reverse=True):
    k = 0
    dev_sum_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    step_size = 50
    t_vocab_list = [k for k, _ in sorted(t_vocab.items(), key=lambda x: x[1])]
    while k < len(valid_srcs):
        dev_loss = 0
        max_s_len = max(len(s) + 1 for s in valid_srcs[k: min(k + step_size, len(valid_srcs))])
        max_t_len = max(len(s) + 1 for s in valid_tgts[k: min(k + step_size, len(valid_srcs))])
        valid_batch_source, valid_batch_target = [], []
        for i in range(k, min(k + step_size, len(valid_srcs))):
            if reverse:
                valid_batch_source.append((valid_srcs[i] + [s_vocab['<EOS>']] * (max_s_len - len(valid_srcs[i])))[::-1])
            else:
                valid_batch_source.append(valid_srcs[i] + [s_vocab['<EOS>']] * (max_s_len - len(valid_srcs[i])))
            valid_batch_target.append(valid_tgts[i] + [t_vocab['<EOS>']] * (max_t_len - len(valid_tgts[i])))

        batch_size = len(valid_batch_source)
        encoder_init_hidden = encoder.init_hidden(batch_size, device)
        xs = torch.tensor(valid_batch_source).to(device)
        ehs = encoder(xs, encoder_init_hidden)

        ys = torch.tensor(valid_batch_target).to(device)
        dhidden = ehs
        pred_words = torch.tensor([[t_vocab['<BOS>']] for _ in range(batch_size)]).to(device)
        pred_seq = []
        i = random.randrange(0, len(valid_batch_source))
        for j in range(max_t_len - 1):
            preds, dhidden = decoder(pred_words, dhidden)
            _, topi = preds.topk(1)
            dev_loss += criterion(preds, ys[:, j])
            pred_words = ys[:, j + 1].view(-1, 1)
            pred_seq.append(topi[i])
        dev_sum_loss += dev_loss.item()
        k += step_size
        print('development data evaluate:')
        print('\t' + ' '.join(t_vocab_list[t] for t in valid_batch_target[i]))
        print('\t' + ' '.join(t_vocab_list[t] if t < len(t_vocab_list) else t_vocab_list[0] for t in pred_seq))
    return dev_sum_loss


def main():
    global device
    args = parse()
    s_vocab = utils.make_vocab(args.train_src, args.vocab_size)
    t_vocab = utils.make_vocab(args.train_tgt, args.vocab_size)
    train_source_seqs, train_target_seqs = [], []
    valid_source_seqs, valid_target_seqs = [], []

    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device('cuda:' + str(args.gpu_id))

    # ファイルを全てID列に変換
    with open(args.train_src, encoding='utf-8') as fin:
        for line in fin:
            train_source_seqs.append(
                [s_vocab[t] if t in s_vocab else s_vocab['<UNK>'] for t in line.strip().split(' ')])
    with open(args.train_tgt, encoding='utf-8') as fin:
        for line in fin:
            train_target_seqs.append(
                [t_vocab[t] if t in t_vocab else t_vocab['<UNK>'] for t in line.strip().split(' ')])
    with open(args.valid_src, encoding='utf-8') as fin:
        for line in fin:
            valid_source_seqs.append(
                [s_vocab[t] if t in s_vocab else s_vocab['<UNK>'] for t in line.strip().split(' ')])
    with open(args.valid_tgt, encoding='utf-8') as fin:
        for line in fin:
            valid_target_seqs.append(
                [t_vocab[t] if t in t_vocab else t_vocab['<UNK>'] for t in line.strip().split(' ')])

    encoder = model.Encoder(args.vocab_size, args.embed_size, args.hidden_size)
    decoder = model.Decoder(args.vocab_size, args.embed_size, args.hidden_size)
    train_losses, valid_losses = train(
        train_source_seqs, train_target_seqs, valid_source_seqs, valid_target_seqs, encoder, decoder,
        s_vocab, t_vocab, args.epochs, args.batch_size, device, args.reverse)

    # テストデータの翻訳に必要な各データを出力
    pickle.dump(s_vocab, open('s_vocab.pkl', 'wb'))
    pickle.dump(t_vocab, open('t_vocab.pkl', 'wb'))
    torch.save(encoder.state_dict(), args.model_prefix + '.enc')
    torch.save(decoder.state_dict(), args.model_prefix + '.dec')

    plt.plot(np.array([i for i in range(1, len(train_losses) + 1)]), train_losses, label='train loss')
    plt.plot(np.array([i for i in range(1, len(valid_losses) + 1)]), valid_losses, label='valid loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.tight_layout()
    plt.savefig('loss_curve.pdf')


if __name__ == '__main__':
    main()
