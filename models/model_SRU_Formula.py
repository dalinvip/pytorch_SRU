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
    def __init__(self, n_in, n_out, dropout=0.0, bias=True):
        super(SRU_Formula_Cell, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.dropout = dropout
        # Linear
        self.x_t = nn.Linear(self.n_in, self.n_out, bias=False)
        self.ft = nn.Linear(self.n_in, self.n_out, bias=bias)
        self.rt = nn.Linear(self.n_in, self.n_out, bias=bias)
        # self.convert_x = nn.Linear(self.n_in, self.n_out, bias=True)
        self.convert_x = self.init_Linear(self.n_in, self.n_out, bias=True)
        self.convert_dim = self.init_Linear(self.n_in, self.n_out, bias=True)
        # dropout
        self.dropout = nn.Dropout(dropout)
        # print(self.x_t)
        # print(self.ft)
        # print(self.rt)
        # print(self.dropout)

    def init_Linear(self, in_fea, out_fea, bias=True):
        linear = nn.Linear(in_features=in_fea, out_features=out_fea, bias=bias)
        return linear

    def forward(self, xt, ct_forward):
        # print(xt.size())
        # print(ct_forward.size())
        # print("////////////////")
        layer = ct_forward.size(0)
        for layers in range(layer):
            # print("aaa", xt.size())
            if xt.size(2) == self.n_out:
                self.convert_dim = SRU_Formula_Cell.init_Linear(self, in_fea=xt.size(2), out_fea=self.n_in)
                xt = self.convert_dim(xt)
            xt, ct = SRU_Formula_Cell.calculate_one_layer(self, xt, ct_forward[layers], layers)
        # for i in range(xt.size(0)):
        #     print("***************")
        #     print(xt[i].size())
        #     x_t = self.x_t(xt[i])
        #     ft = F.sigmoid(self.ft(xt[i]))
        #     rt = F.sigmoid(self.rt(xt[i]))
        #     print("x_t", x_t.size())
        #     print("ft", ft.size())
        #     print("rt", rt.size())
        #     self.convert_dim = self.init_Linear(in_fea=ct_forward.size(0), out_fea=ft.size(0), bias=True)
        #     ct_forward = self.convert_dim(ct_forward.permute(1, 2, 0))
        #     # print("ct_con", ct_forward.size())
        #     ct = torch.add(torch.mul(ft, ct_forward), torch.mul((1 - ft), x_t))
        #     # print("ct", ct.size())
        #     con_xt = self.convert_x(xt)
        #     ht = torch.add(torch.mul(rt, F.tanh(ct)), torch.mul((1 - rt), con_xt))

        if self.dropout is not None:
            ht = self.dropout(xt)
            ct = self.dropout(ct)
        return ht, ct

    def calculate_one_layer(self, xt, ct_forward, layer):
        # print(xt.size())
        # print(ct_forward.size())
        # init c
        ct = ct_forward
        ht_list = []
        for i in range(xt.size(0)):
            # print("***************")
            # print(xt[i].size())
            # print(layer)
            x_t = self.x_t(xt[i])
            ft = F.sigmoid(self.ft(xt[i]))
            rt = F.sigmoid(self.rt(xt[i]))
            # print("x_t", x_t.size())
            # print("ft", ft.size())
            # print("rt", rt.size())
            self.convert_dim = self.init_Linear(in_fea=ct_forward.size(0), out_fea=ft.size(0), bias=True)
            # ct_forward = self.convert_dim(ct_forward.permute(1, 2, 0))
            # print("ct_con", ct_forward.size())
            ct = torch.add(torch.mul(ft, ct), torch.mul((1 - ft), x_t))
            # print("ct", ct.size())
            con_xt = self.convert_x(xt[i])
            ht = torch.add(torch.mul(rt, F.tanh(ct)), torch.mul((1 - rt), con_xt))
            ht_list.append(ht.unsqueeze(0))
            # print(len(ht_list))
            # print("ct", ct.size())
            # print("ht", ht.size())
        # for i in range(len(ht_list) - 1):
        ht = torch.cat(ht_list, 0)
        # print(ht.size())
        if self.dropout is not None:
            ht = self.dropout(ht)
            ct = self.dropout(ct)
        return ht, ct



class SRU_Formula(nn.Module):
    def __init__(self, args):
        super(SRU_Formula, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
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

        self.sru = SRU_Formula_Cell(n_in=D, n_out=self.hidden_dim, dropout=args.dropout, bias=True)
        print(self.sru)

        # if args.init_weight:
        #     print("Initing W .......")
        #     init.xavier_normal(self.sru.all_weights[0][0], gain=np.sqrt(args.init_weight_value))
        #     init.xavier_normal(self.sru.all_weights[0][1], gain=np.sqrt(args.init_weight_value))
        #     init.xavier_normal(self.sru.all_weights[1][0], gain=np.sqrt(args.init_weight_value))
        #     init.xavier_normal(self.sru.all_weights[1][1], gain=np.sqrt(args.init_weight_value))
        if args.cuda is True:
            self.hidden2label = nn.Linear(self.hidden_dim, C).cuda()
            self.hidden = self.init_hidden(self.num_layers, args.batch_size).cuda()
        else:
            self.hidden2label = nn.Linear(self.hidden_dim, C)
            self.hidden = self.init_hidden(self.num_layers, args.batch_size)
        # print("self.hidden", self.hidden)

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
        # x = x.view(len(x), x.size(1), -1)
        # x = embed.view(len(x), embed.size(1), -1)
        # print(x.size())
        # self.hidden = SRU_Formula.init_hidden_c(self, x.size(0), x.size(1))
        # print("for", self.hidden)
        sru_out, self.hidden = self.sru(x, self.hidden)
        # print("after", self.hidden)

        sru_out = torch.transpose(sru_out, 0, 1)
        sru_out = torch.transpose(sru_out, 1, 2)
        sru_out = F.tanh(sru_out)
        sru_out = F.max_pool1d(sru_out, sru_out.size(2)).squeeze(2)
        sru_out = F.tanh(sru_out)

        logit = self.hidden2label(sru_out)

        return logit