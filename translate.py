# -*- coding: utf-8 -*-

import argparse
from io import open
import pickle
import sys
import torch

import models

device = torch.device('cpu')


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-src')
    parser.add_argument('-output')
    parser.add_argument('--gpu_id', '-g', default=-1, type=int)
    parser.add_argument('--s_vocab', '-sv', default='s_vocab.pkl')
    parser.add_argument('--t_vocab', '-tv', default='t_vocab.pkl')
    parser.add_argument('--model_prefix', '-mf', default='encdec')
    parser.add_argument('--vocab_size', '-vs', default=50000, type=int)
    parser.add_argument('--embed_size', '-es', default=256, type=int)
    parser.add_argument('--hidden_size', '-hs', default=256, type=int)
    parser.add_argument('--model_type', '-mt', default='EncDec', help='Model type (`EncDec` or `Attn`)')
    parser.add_argument('--reverse', '-r', action='store_true', default=False)

    args = parser.parse_args()
    return args


def translate(src, model, s_vocab, t_vocab, output, device, max_len, reverse=False):
    t_vocab_list = [k for k, _ in sorted(t_vocab.items(), key=lambda x: x[1])]
    with open(src, encoding='utf-8') as fin, open(output, 'w', encoding='utf-8') as fout:
        input_sequences = []
        for i, line in enumerate(fin):
            input_sequences.append([s_vocab[t] if t in s_vocab else s_vocab['<UNK>'] for t in line.strip().split(' ')])
            if (i + 1) % 50 != 0:
                continue
            max_s_len = max(len(s) for s in input_sequences)
            for j in range(len(input_sequences)):
                if reverse:
                    input_sequences[j] = (input_sequences[j] +
                                          [s_vocab['<EOS>']] * (max_s_len - len(input_sequences[j])))[::-1]
                else:
                    input_sequences[j] = (input_sequences[j] +
                                          [s_vocab['<EOS>']] * (max_s_len - len(input_sequences[j])))

            xs = torch.tensor(input_sequences, device=device).t().contiguous()
            pred_seqs = model.predict(xs, device, max_len)
            for pred_seq in pred_seqs:
                print(' '.join(t_vocab_list[t] if 0 <= t < len(t_vocab_list) else t_vocab_list[0]
                               for t in pred_seq), file=fout)
            input_sequences
            print('\r%d sentence translated' % (i + 1), end='')

        if input_sequences:
            max_s_len = max(len(s) for s in input_sequences)
            for j in range(len(input_sequences)):
                if reverse:
                    input_sequences[j] = (input_sequences[j] +
                                          [s_vocab['<EOS>']] * (max_s_len - len(input_sequences[j])))[::-1]
                else:
                    input_sequences[j] = (input_sequences[j] +
                                          [s_vocab['<EOS>']] * (max_s_len - len(input_sequences[j])))

            xs = torch.tensor(input_sequences, device=device).t().contiguous()
            pred_seqs = model.predict(xs, device, max_len)
            for pred_seq in pred_seqs:
                print(' '.join(t_vocab_list[t] if 0 <= t < len(t_vocab_list) else t_vocab_list[0]
                               for t in pred_seq), file=fout)
    print('\nTranslating is finished!')


def main():
    global device
    args = parse()
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device('cuda:' + str(args.gpu_id))
    s_vocab = pickle.load(open(args.s_vocab, 'rb'))
    t_vocab = pickle.load(open(args.t_vocab, 'rb'))
    vs, es, hs = args.vocab_size, args.embed_size, args.hidden_size
    if args.model_type == 'EncDec':
        model = models.EncoderDecoder(
            s_vocab_size=vs, t_vocab_size=vs, hidden_size=hs, embed_size=es, weight_decay=1e-5
        ).to(device)
    elif args.model_type == 'Attn':
        model = models.AttentionSeq2Seq(
            s_vocab_size=vs, t_vocab_size=vs, embed_size=es, hidden_size=hs,
            num_s_layers=2, bidirectional=True, weight_decay=1e-5
        ).to(device)
    else:
        sys.stderr.write('%s is not found. Model type is `EncDec` or `Attn`.' % args.model_type)
    model.load_state_dict(torch.load(args.model_prefix + '.model'))
    translate(args.src, model, s_vocab, t_vocab, args.output, device, 100, reverse=args.reverse)


if __name__ == '__main__':
    main()
