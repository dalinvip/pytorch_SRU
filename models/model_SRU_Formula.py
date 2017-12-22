import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import torch.nn.init as init
import hyperparams
torch.manual_seed(hyperparams.seed_num)
random.seed(hyperparams.seed_num)


class SRU_Formula_Cell(nn.Module):
    def __init__(self, args, n_in, n_out, layer_numbers=1, dropout=0.0, bias=True):
        super(SRU_Formula_Cell, self).__init__()
        self.args = args
        self.layer_numbers = layer_numbers
        self.n_in = n_in
        self.n_out = n_out
        self.dropout = dropout
        # Linear
        self.x_t = nn.Linear(self.n_in, self.n_out, bias=False)
        self.ft = nn.Linear(self.n_in, self.n_out, bias=bias)
        self.rt = nn.Linear(self.n_in, self.n_out, bias=bias)
        # self.convert_x = nn.Linear(self.n_in, self.n_out, bias=True)
        self.convert_x = self.init_Linear(self.n_in, self.n_out, bias=True)
        self.convert_x_layer = self.init_Linear(self.n_out, self.args.embed_dim, bias=True)
        self.convert_dim = self.init_Linear(self.n_in, self.n_out, bias=True)
        # dropout
        self.dropout = nn.Dropout(dropout)


    def init_Linear(self, in_fea, out_fea, bias=True):
        linear = nn.Linear(in_features=in_fea, out_features=out_fea, bias=bias)
        if self.args.cuda is True:
            return linear.cuda()
        else:
            return linear

    def forward(self, xt, ct_forward):
        layer = self.layer_numbers
        for layers in range(layer):
            if xt.size(2) == self.n_out:
                xt = self.convert_x_layer(xt)
            # xt = self.convert_x(xt)
            xt, ct = SRU_Formula_Cell.calculate_one_layer(self, xt, ct_forward[layers])
        if self.dropout is not None:
            ht = self.dropout(xt)
            ct = self.dropout(ct)
        if self.args.cuda is True:
            return ht.cuda(), ct.cuda()
        else:
            return ht, ct

    def calculate_one_layer(self, xt, ct_forward):
        # print(xt.size())
        # print(ct_forward.size())
        # init c
        ct = ct_forward
        ht_list = []
        for i in range(xt.size(0)):
            x_t = self.x_t(xt[i])
            if self.args.cuda is True:
                ft = F.sigmoid(self.ft(xt[i]).cuda()).cuda()
            else:
                ft = F.sigmoid(self.ft(xt[i]))
            rt = F.sigmoid(self.rt(xt[i]))
            self.convert_dim = self.init_Linear(in_fea=ct_forward.size(0), out_fea=ft.size(0), bias=True)
            ct = torch.add(torch.mul(ft, ct), torch.mul((1 - ft), x_t))
            con_xt = self.convert_x(xt[i])
            ht = torch.add(torch.mul(rt, F.tanh(ct)), torch.mul((1 - rt), con_xt))
            ht_list.append(ht.unsqueeze(0))
        ht = torch.cat(ht_list, 0)
        if self.args.cuda is True:
            return ht.cuda(), ct.cuda()
        else:
            return ht, ct


class SRU_Formula(nn.Module):
    def __init__(self, args):
        super(SRU_Formula, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        print("layers", self.num_layers)
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        self.dropout = nn.Dropout(args.dropout)
        self.dropout_embed = nn.Dropout(args.dropout_embed)
        self.embed = nn.Embedding(V, D)
        if args.fix_Embedding is True:
            self.embed.weight.requires_grad = False
        if args.word_Embedding:
            pretrained_weight = np.array(args.pretrained_weight)
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))

        self.sru = SRU_Formula_Cell(self.args, n_in=D, n_out=self.hidden_dim, layer_numbers=self.num_layers,
                                    dropout=args.dropout, bias=True)
        print(self.sru)
        if self.args.cuda is True:
            self.sru.cuda()

        if args.cuda is True:
            self.hidden2label = nn.Linear(self.hidden_dim, C).cuda()
            self.hidden = self.init_hidden(self.num_layers, args.batch_size).cuda()
        else:
            self.hidden2label = nn.Linear(self.hidden_dim, C)
            self.hidden = self.init_hidden(self.num_layers, args.batch_size)

    def init_hidden(self, num_layers, batch_size):
        # the first is the hidden h
        # the second is the cell  c
        if self.args.cuda is True:
            return Variable(torch.zeros(num_layers, batch_size, self.hidden_dim)).cuda()
        else:
            return Variable(torch.zeros(num_layers, batch_size, self.hidden_dim))

    def init_hidden_c(self, length, batch_size):
        # the first is the hidden h
        # the second is the cell  c
        if self.args.cuda is True:
            return Variable(torch.zeros(length, batch_size, self.hidden_dim)).cuda()
        else:
            return Variable(torch.zeros(length, batch_size, self.hidden_dim))

    def forward(self, x):
        x = self.embed(x)
        x = self.dropout_embed(x)
        sru_out, self.hidden = self.sru(x, self.hidden)

        sru_out = torch.transpose(sru_out, 0, 1)
        sru_out = torch.transpose(sru_out, 1, 2)
        sru_out = F.tanh(sru_out)
        sru_out = F.max_pool1d(sru_out, sru_out.size(2)).squeeze(2)
        sru_out = F.tanh(sru_out)

        logit = self.hidden2label(sru_out)

        return logit