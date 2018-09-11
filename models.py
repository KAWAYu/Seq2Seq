# -*- coding: utf-8 -*-

import random
import torch
import torch.nn as nn

import nn_blocks



class EncoderDecoder(nn.Module):
    def __init__(self, **kwargs):
        super(EncoderDecoder, self).__init__()
        self.source_vocab_size = kwargs['s_vocab_size']
        self.target_vocab_size = kwargs['t_vocab_size']
        self.hidden_size = kwargs['hidden_size']
        self.embed_size = kwargs['embed_size']
        self.num_source_layers = kwargs['num_s_layers'] if 'num_s_layers' in kwargs else 1
        self.num_target_layers = kwargs['num_t_layers'] if 'num_t_layers' in kwargs else 1
        self.weight_decay = kwargs['weight_decay']

        self.encoder = nn_blocks.Encoder(
            self.source_vocab_size, self.embed_size, self.hidden_size, self.num_source_layers)
        self.decoder = nn_blocks.Decoder(
            self.target_vocab_size, self.embed_size, self.hidden_size, self.num_target_layers)
        self.optim = torch.optim.Adam(self.parameters(), weight_decay=self.weight_decay)

    def forward(self, xs, ys, criterion, device):
        """
        encoder decoder model
        :param xs: input tensor, [x_1, x_2, ..., x_m]
        :param ys: target tensor, [<BOS>, y_1, y_2, ..., y_n]
        :param criterion: criterion instance
        :param device: device(cpu or gpu)
        :return: loss
        """
        loss = 0
        self.optim.zero_grad()
        batch_size = xs.size(1)
        encoder_init_hidden = self.encoder.init_hidden(batch_size, device)
        _, ehs = self.encoder(xs, encoder_init_hidden)

        dhidden = ehs
        for j in range(len(ys[0]) - 1):
            prev_words = ys[:, j].unsqueeze(0)
            preds, dhidden = self.decoder(prev_words, dhidden)
            _, topi = preds.topk(1)
            loss += criterion(preds.view(-1, self.target_vocab_size), ys[:, j + 1])
        loss.backward()
        self.optim.step()
        return loss

    def predict(self, xs, device, max_len, BOS_token=2, EOS_token=1):
        batch_size = xs.size(1)
        encoder_init_hidden = self.encoder.init_hidden(batch_size, device)
        _, ehs = self.encoder(xs, encoder_init_hidden)

        dhidden = ehs
        pred_seqs = [[] for _ in range(batch_size)]
        prev_words = torch.tensor([[BOS_token for _ in range(batch_size)]], device=device)
        for _ in range(max_len):
            preds, dhidden = self.decoder(prev_words, dhidden)
            _, topi = preds.topk(1)
            for k in range(len(pred_seqs)):
                pred_seqs[k].append(topi[k])
            prev_words = topi.view(1, -1).detach()

            if all(topii == EOS_token for topii in topi):
                break

        _pred_seqs = [pred_seq[:pred_seq.index(EOS_token)] if EOS_token in pred_seq
                      else pred_seq for pred_seq in pred_seqs]
        return _pred_seqs


class AttentionSeq2Seq(nn.Module):
    def __init__(self, **kwargs):
        super(AttentionSeq2Seq, self).__init__()
        self.source_vocab_size = kwargs['s_vocab_size']
        self.target_vocab_size = kwargs['t_vocab_size']
        self.hidden_size = kwargs['hidden_size']
        self.embed_size = kwargs['embed_size']
        self.num_source_layers = kwargs['num_s_layers'] if 'num_s_layers' in kwargs else 1
        self.num_target_layers = kwargs['num_t_layers'] if 'num_t_layers' in kwargs else 1
        self.is_bidirectional = kwargs['bidirectional']
        self.weight_decay = kwargs['weight_decay']

        self.encoder = nn_blocks.Encoder(self.source_vocab_size, self.embed_size, self.hidden_size,
                                         self.num_source_layers, self.is_bidirectional)
        self.attn_decoder = nn_blocks.AttentionDecoder(self.target_vocab_size, self.embed_size, self.hidden_size,
                                                       self.num_target_layers)
        self.optim = torch.optim.Adam(self.parameters(), weight_decay=self.weight_decay)

    def forward(self, xs, ys, criterion, device):
        loss = 0
        self.optim.zero_grad()

        batch_size = xs.size(1)
        encoder_init_hidden = self.encoder.init_hidden(batch_size, device)
        dhidden = self.attn_decoder.init_hidden(batch_size, device)

        ehs, _ = self.encoder(xs, encoder_init_hidden)

        i = random.randrange(0, batch_size)
        seqs = []
        for j in range(len(ys[0]) - 1):
            prev_words = ys[:, j].unsqueeze(0).to(device)
            preds, dhidden = self.attn_decoder(prev_words, dhidden, ehs)
            _, topi = preds.topk(1)
            seqs.append(topi[i].item())
            loss += criterion(preds, ys[:, j + 1])
        loss.backward()
        self.optim.step()
        return loss, i, seqs

    def predict(self, xs, device, max_len, BOS_token=2, EOS_token=1):
        batch_size = xs.size(1)
        encoder_init_hidden = self.encoder.init_hidden(batch_size, device)
        ehs, _ = self.encoder(xs, encoder_init_hidden)

        dhidden = self.attn_decoder.init_hidden(batch_size, device)
        prev_words = torch.tensor([[BOS_token for _ in range(batch_size)]], device=device)
        pred_seqs = [[] for _ in range(batch_size)]

        for _ in range(max_len):
            preds, dhidden = self.attn_decoder(prev_words, dhidden, ehs)
            _, topi = preds.topk(1)
            for k in range(len(pred_seqs)):
                pred_seqs[k].append(topi[k])

            if all(topii == EOS_token for topii in topi):
                break
            prev_words = topi.view(1, -1).detach()
        _pred_seqs = [pred_seq[:pred_seq.index(EOS_token)] if EOS_token in pred_seq
                      else pred_seq for pred_seq in pred_seqs]
        return _pred_seqs
