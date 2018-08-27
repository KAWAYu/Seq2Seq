# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, bidirectional=False):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.is_bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.e2h = nn.Linear(embed_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, xs, prev_hidden):
        h = torch.tanh(self.e2h(self.embedding(xs)))  # (batch_size, seq_len) => (batch_size, seq_len, hidden_size)
        _, hidden = self.gru(h, prev_hidden)
        return hidden

    def init_hidden(self, batch_size, device):
        direction = 2 if self.is_bidirectional else 1
        return torch.zeros(self.num_layers * direction, batch_size, self.hidden_size, device=device)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.e2h = nn.Linear(embed_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.h2e = nn.Linear(hidden_size, embed_size)
        self.e2y = nn.Linear(embed_size, vocab_size)

    def forward(self, prev_ys, prev_hidden):
        h = torch.tanh(self.e2h(self.embedding(prev_ys)))
        out, hidden = self.gru(h, prev_hidden)
        y_distribution = self.e2y(torch.tanh(self.h2e(out.squeeze(1))))
        return y_distribution, hidden


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.e2h = nn.Linear(embed_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.c2h = nn.Linear(hidden_size * 2, hidden_size)
        self.h2e = nn.Linear(hidden_size, embed_size)
        self.e2v = nn.Linear(embed_size, vocab_size)

    def forward(self, prev_ys, prev_hidden, ehs):
        batch_size = prev_ys.size(0)
        h = torch.tanh(self.e2h(self.embedding(prev_ys)))
        out, hidden = self.gru(h, prev_hidden)
        attention_weight = torch.exp(torch.tanh(torch.sum(ehs * hidden, dim=2)))
        attention = attention_weight / torch.sum(attention_weight, dim=0)
        context_vector = torch.sum(ehs * attention.view(-1, batch_size, 1), dim=0)
        output = torch.tanh(self.c2h(torch.cat((hidden.squeeze(0), context_vector), dim=1)))
        output = self.e2v(torch.tanh(self.h2e(output)))
        return output, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class ConvBlock(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=5, residual=True):
        raise NotImplementedError('Sorry, this implementation is future work.')
        super(ConvBlock, self).__init__()
        self.hidden_size = hidden_size
        self.is_residual = residual

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.e2h = nn.Linear(embed_size, hidden_size)
        self.conv = nn.Conv2d(1, 1, (3, hidden_size), 1, 1)

    def forward(self, xs, ys, device):
        pass
