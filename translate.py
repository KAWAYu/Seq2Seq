# -*- coding: utf-8 -*-

import argparse
from io import open
import pickle
import torch

import model

device = torch.device('cpu')


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-src')
    parser.add_argument('-output')
    parser.add_argument('--gpu_id', '-g', default=-1, type=int)
    parser.add_argument('--s_vocab', '-sv', default='s_vocab.pkl')
    parser.add_argument('--t_vocab', '-tv', default='t_vocab.pkl')
    parser.add_argument('--model_prefix', '-mf', default='model')
    parser.add_argument('--vocab_size', '-vs', default=50000)
    parser.add_argument('--embed_size', '-es', default=256)
    parser.add_argument('--hidden_size', '-hs', default=256)
    parser.add_argument('--reverse', '-r', action='store_true', default=False)

    args = parser.parse_args()
    return args


def translate(src, encoder, decoder, s_vocab, t_vocab, output, device, max_len, reverse=False):
    t_vocab_list = [k for k, _ in sorted(t_vocab.items(), key=lambda x: x[1])]
    with open(src, encoding='utf-8') as fin, open(output, 'w', encoding='utf-8') as fout:
        input_sequences = []
        for i, line in enumerate(fin):
            input_sequences.append([s_vocab[t] if t in s_vocab else s_vocab['<UNK>'] for t in line.strip().split(' ')])
            if (i + 1) % 100 != 0:
                continue
            max_s_len = max(len(s) for s in input_sequences)
            for j in range(len(input_sequences)):
                if reverse:
                    input_sequences[j] = (input_sequences[j] +
                                          [s_vocab['<EOS>']] * (max_s_len - len(input_sequences[j])))[::-1]
                else:
                    input_sequences[j] = (input_sequences[j] +
                                          [s_vocab['<EOS>']] * (max_s_len - len(input_sequences[j])))

            encoder_hidden = encoder.init_hidden(100, device)
            xs = torch.tensor(input_sequences).to(device)
            ehs = encoder(xs, encoder_hidden)

            dhidden = ehs
            pred_words = torch.tensor([[t_vocab['<BOS>']] for _ in range(100)]).to(device)
            pred_seqs = [[] for _ in range(100)]
            for _ in range(max_len):
                preds, dhidden = decoder(pred_words, dhidden)
                _, topi = preds.topk(1)
                for k in range(len(pred_seqs)):
                    pred_seqs[k].append(topi[k])

                if all(topii == t_vocab['<EOS>'] for topii in topi):
                    break

            for pred_seq in pred_seqs:
                _pred_seq = []
                for p in pred_seq:
                    if p == t_vocab['<EOS>']:
                        break
                    _pred_seq.append(p)
                print(' '.join(t_vocab_list[t] if 0 <= t < len(t_vocab_list) else t_vocab_list[0]
                               for t in _pred_seq), file=fout)
            input_sequences = []
            print('%d sentence translated' % (i + 1))

        if input_sequences:
            max_s_len = max(len(s) for s in input_sequences)
            for j in range(len(input_sequences)):
                input_sequences[j] = input_sequences[j] + [s_vocab['<EOS>']] * (max_s_len - len(input_sequences[j]))

            encoder_hidden = encoder.init_hidden(len(input_sequences), device)
            xs = torch.tensor(input_sequences).to(device)
            ehs = encoder(xs, encoder_hidden)

            dhidden = ehs
            pred_words = torch.tensor([[t_vocab['<BOS>']] for _ in range(len(input_sequences))]).to(device)
            pred_seqs = [[] for _ in range(len(input_sequences))]
            for _ in range(max_len):
                preds, dhidden = decoder(pred_words, dhidden)
                _, topi = preds.topk(1)
                for i in range(len(pred_seqs)):
                    pred_seqs[i].append(topi[i])

                if all(topii == t_vocab['<EOS>'] for topii in topi):
                    break

            for pred_seq in pred_seqs:
                _pred_seq = []
                for p in pred_seq:
                    if p == t_vocab['<EOS>']:
                        break
                    _pred_seq.append(p)
                print(' '.join(t_vocab_list[t] if 0 <= t < len(t_vocab_list) else t_vocab_list[0]
                               for t in _pred_seq), file=fout)


def main():
    global device
    args = parse()
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device('cuda:' + str(args.gpu_id))
    s_vocab = pickle.load(open(args.s_vocab, 'rb'))
    t_vocab = pickle.load(open(args.t_vocab, 'rb'))
    vs, es, hs = args.vocab_size, args.embed_size, args.hidden_size
    encoder = model.Encoder(vs, es, hs)
    decoder = model.Decoder(vs, es, hs)
    encoder.load_state_dict(torch.load(args.model_prefix + '.enc'))
    decoder.load_state_dict(torch.load(args.model_prefix + '.dec'))
    translate(args.src, encoder, decoder, s_vocab, t_vocab, args.output, device, 100, reverse=args.reverse)


if __name__ == '__main__':
    main()
