# -*- coding: utf-8 -*-

from io import open


def make_vocab(filepath, max_size, init_vocab={}):
    vocab = {'<UNK>': 0, '<EOS>': 1, '<BOS>': 2}
    vocab_count = {}
    with open(filepath, encoding='utf-8') as fin:
        for line in fin:
            tokens = line.strip().split(' ')
            for token in tokens:
                if token in vocab_count:
                    vocab_count[token] += 1
                else:
                    vocab_count[token] = 1
    for k, _ in sorted(vocab_count.items(), key=lambda x: -x[1]):
        vocab[k] = len(vocab)
        if len(vocab) >= max_size:
            break
    return vocab


def sorted_batch_generator(src, tgt, batch_size):
    """
    バッチ内の長さをできるだけ揃えるために長さ順にソートしてバッチ分だけ生成するジェネレータ
    :param src: 原言語側
    :param tgt: 目的言語
    :param batch_size: バッチサイズ
    :return (src_batch, tgt_batch): バッチ分の文対
    """
    src_tgt = [(s, t) for s, t in zip(src, tgt)]
    src_tgt = sorted(src_tgt, key=lambda x: len(x[0]))
    for k in range(len(src) // batch_size + (len(src) % batch_size)):
        src_batch = [src_tgt[i][0] for i in range(k * batch_size, min(k * batch_size + batch_size, len(src)))]
        tgt_batch = [src_tgt[i][1] for i in range(k * batch_size, min(k * batch_size + batch_size, len(src)))]
        yield src_batch, tgt_batch


def sorted_padded_batch(src, tgt, batch_size, pad_id):
    src_batches, tgt_batches = [], []
    for src_batch, tgt_batch in sorted_batch_generator(src, tgt, batch_size):
        max_s_len = max(len(s) + 1 for s in src_batch)
        max_t_len = max(len(t) + 1 for t in tgt_batch)
        for i in range(len(src_batch)):
            src_batch[i] = src_batch[i] + [pad_id] * (max_s_len - len(src_batch[i]))
            tgt_batch[i] = tgt_batch[i] + [pad_id] * (max_t_len - len(tgt_batch[i]))
        src_batches.append(src_batch)
        tgt_batches.append(tgt_batch)
    return src_batches, tgt_batches


if __name__ == '__main__':
    print('This is a utility box.')
