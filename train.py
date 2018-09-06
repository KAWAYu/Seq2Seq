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
import sys

import models
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
    parser.add_argument('--model_prefix', '-mf', default='encdec', help='prefix of model name')
    parser.add_argument('--model_type', '-mt', default='EncDec', help='Model type (`EncDec` or `Attn`)')
    parser.add_argument('--reverse', '-r', action='store_true', default=False, help='reverse order of input sequences')

    args = parser.parse_args()
    return args


def train(train_srcs, train_tgts, valid_srcs, valid_tgts, model, s_vocab, t_vocab, num_epochs,
          batch_size, device, reverse=True):
    """
    訓練する関数
    :param train_srcs: 原言語の訓練データ
    :param train_tgts: 目的言語の訓練データ
    :param valid_srcs: 原言語の開発データ
    :param valid_tgts: 目的言語の開発データ
    :param model: ニューラルネットのモデル
    :param s_vocab: 原言語側の語彙辞書 (語彙 -> ID)
    :param t_vocab: 目的言語の語彙辞書 (語彙 -> ID)
    :param num_epochs: エポック数
    :param batch_size: バッチ数
    :param device: CPUかGPUの指定(torch.device)
    :param reverse: 入力を逆順にするかのフラグ
    :return tuple(train_loss, dev_loss): 訓練データでの各エポックでのロス、開発データでの各エポックでのロス
    """

    criterion = torch.nn.CrossEntropyLoss()  # 損失関数

    # t_vocab_list = [k for k, _ in sorted(t_vocab.items(), key=lambda x: x[1])]
    train_losses, dev_losses = [], []

    for e in range(1, num_epochs + 1):
        print('Epoch %d start...' % e)
        total_loss = 0
        indexes = [i for i in range(len(train_srcs))]  # 0..len(train_srcs)のインデックスのリストを作成し、
        random.shuffle(indexes)                        # 順番をシャッフル
        k = 0  # 何文処理をしたかを格納するカウンタ変数
        while k < len(indexes):
            batch_idx = indexes[k: min(k + batch_size, len(indexes))]  # batch_size分のインデックスを取り出し
            batch_t_s, batch_t_t = [], []
            # バッチ処理用の文対をまとめる
            for idx in batch_idx:
                batch_t_s.append(train_srcs[idx])
                batch_t_t.append(train_tgts[idx])

            batch_src_len = [len(s) + 1 for s in batch_t_s]  # 各バッチの長さ（<EOS>のインデックス + 1）を記録
            # TODO: 各バッチの長さを使って<EOS>タグが入力されたときの隠れ層を取得できるようにしたい
            max_s_len = max(batch_src_len)
            max_t_len = max(len(s) + 1 for s in batch_t_t)
            # バッチの中で一番長い文に合わせてpadding
            for i in range(len(batch_t_s)):
                if reverse:
                    batch_t_s[i] = (batch_t_s[i] + [s_vocab['<EOS>']] * (max_s_len - len(batch_t_s[i])))[::-1]
                else:
                    batch_t_s[i] = (batch_t_s[i] + [s_vocab['<EOS>']] * (max_s_len - len(batch_t_s[i])))
                batch_t_t[i] = [t_vocab['<BOS>']] + batch_t_t[i] + [t_vocab['<EOS>']] * (max_t_len - len(batch_t_t[i]))

            # 訓練は「入力シーケンス」と「出力シーケンス」を渡すだけ（中で重みの更新までする）
            xs = torch.tensor(batch_t_s).to(device)
            ys = torch.tensor(batch_t_t).to(device)
            batch_loss = model(xs, ys, criterion, device)

            total_loss += batch_loss.item()
            k += len(batch_idx)
            print('\r%d sentences was learned, loss %.4f' % (k, batch_loss.item()), end='')
        print()
        train_losses.append(total_loss / len(train_srcs))
        dev_loss = dev_evaluate(valid_srcs, valid_tgts, model, s_vocab, t_vocab, device, reverse=reverse)
        dev_losses.append(dev_loss / len(valid_srcs))
        print('train loss avg: %.6f, valid loss avg: %.6f' % (total_loss / len(train_srcs), dev_loss / len(valid_srcs)))
    return train_losses, dev_losses


def dev_evaluate(valid_srcs, valid_tgts, model, s_vocab, t_vocab, device, reverse=True):
    k = 0
    dev_sum_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    step_size = 50
    while k < len(valid_srcs):
        max_s_len = max(len(s) + 1 for s in valid_srcs[k: min(k + step_size, len(valid_srcs))])
        max_t_len = max(len(s) + 1 for s in valid_tgts[k: min(k + step_size, len(valid_srcs))])
        valid_batch_source, valid_batch_target = [], []
        for i in range(k, min(k + step_size, len(valid_srcs))):
            if reverse:
                valid_batch_source.append((valid_srcs[i] + [s_vocab['<EOS>']] * (max_s_len - len(valid_srcs[i])))[::-1])
            else:
                valid_batch_source.append(valid_srcs[i] + [s_vocab['<EOS>']] * (max_s_len - len(valid_srcs[i])))
            valid_batch_target.append([t_vocab['<BOS>']] + valid_tgts[i] + [t_vocab['<EOS>']] * (max_t_len - len(valid_tgts[i])))

        xs = torch.tensor(valid_batch_source).to(device)
        ys = torch.tensor(valid_batch_target).to(device)
        batch_loss = model(xs, ys, criterion, device)

        dev_sum_loss += batch_loss.item()
        k += step_size
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

    if args.model_type == 'EncDec':
        model = models.EncoderDecoder(s_vocab_size=args.vocab_size, t_vocab_size=args.vocab_size,
                                      embed_size=args.embed_size, hidden_size=args.hidden_size,
                                      weight_decay=1e-5).to(device)
    elif args.model_type == 'Attn':
        model = models.AttentionSeq2Seq(s_vocab_size=args.vocab_size, t_vocab_size=args.vocab_size,
                                        embed_size=args.embed_size, hidden_size=args.hidden_size,
                                        num_s_layers=2, bidirectional=True, weight_decay=1e-5).to(device)
    else:
        sys.stderr.write('%s is not found. Model type is `EncDec` or `Attn`.' % args.model_type)

    train_losses, valid_losses = train(
        train_source_seqs, train_target_seqs, valid_source_seqs, valid_target_seqs, model,
        s_vocab, t_vocab, args.epochs, args.batch_size, device, args.reverse)

    # テストデータの翻訳に必要な各データを出力
    pickle.dump(s_vocab, open('s_vocab.pkl', 'wb'))
    pickle.dump(t_vocab, open('t_vocab.pkl', 'wb'))
    torch.save(model.state_dict(), args.model_prefix + '.model')

    plt.plot(np.array([i for i in range(1, len(train_losses) + 1)]), train_losses, label='train loss')
    plt.plot(np.array([i for i in range(1, len(valid_losses) + 1)]), valid_losses, label='valid loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.tight_layout()
    plt.savefig('loss_curve.pdf')


if __name__ == '__main__':
    main()
