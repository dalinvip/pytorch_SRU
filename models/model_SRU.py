import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import torch.nn.init as init
from models import cuda_functional as MF
import hyperparams
torch.manual_seed(hyperparams.seed_num)
random.seed(hyperparams.seed_num)


class SRU(nn.Module):
    def __init__(self, args):
        super(SRU, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        self.dropout = nn.Dropout(args.dropout)
        self.dropout_embed = nn.Dropout(args.dropout_embed)
        if args.max_norm is not None:
            print("max_norm = {} ".format(args.max_norm))
            self.embed = nn.Embedding(V, D, max_norm=args.max_norm, scale_grad_by_freq=True)
        else:
            print("max_norm = {} |||||".format(args.max_norm))
            self.embed = nn.Embedding(V, D, scale_grad_by_freq=True)
        if args.fix_Embedding is True:
            self.embed.weight.requires_grad = False
        if args.word_Embedding:
            pretrained_weight = np.array(args.pretrained_weight)
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))

        self.sru = MF.SRU(input_size=D, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                          dropout=self.args.dropout, bidirectional=False)
        print(self.sru)

        # if args.init_weight:
        #     print("Initing W .......")
        #     init.xavier_normal(self.sru.all_weights[0][0], gain=np.sqrt(args.init_weight_value))
        #     init.xavier_normal(self.sru.all_weights[0][1], gain=np.sqrt(args.init_weight_value))
        #     init.xavier_normal(self.sru.all_weights[1][0], gain=np.sqrt(args.init_weight_value))
        #     init.xavier_normal(self.sru.all_weights[1][1], gain=np.sqrt(args.init_weight_value))
        self.hidden2label = nn.Linear(self.hidden_dim, C)
        self.hidden = self.init_hidden(self.num_layers, args.batch_size)
        print("self.hidden", self.hidden)

    def init_hidden(self, num_layers, batch_size):
        # the first is the hidden h
        # the second is the cell  c
        return (Variable(torch.zeros(num_layers, batch_size, self.hidden_dim)),
                Variable(torch.zeros(num_layers, batch_size, self.hidden_dim)))

    def forward(self, x):
        x = self.embed(x)
        x = self.dropout_embed(x)
        # x = x.view(len(x), x.size(1), -1)
        # x = embed.view(len(x), embed.size(1), -1)
        sru_out, self.hidden = self.sru(x)

        # sru_out = sru_out[-1]
        #
        sru_out = torch.transpose(sru_out, 0, 1)
        sru_out = torch.transpose(sru_out, 1, 2)
        sru_out = F.tanh(sru_out)
        sru_out = F.max_pool1d(sru_out, sru_out.size(2)).squeeze(2)
        sru_out = F.tanh(sru_out)
        # sru_out = self.dropout(sru_out)

        logit = self.hidden2label(sru_out)

        return logit