import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Controller(torch.nn.Module):
    def __init__(self, num_edge, num_ops, lstm_size=64, lstm_num_layers=1, tanh_constant=1.10, temperature=5.0):
        torch.nn.Module.__init__(self)
        # TODO
        self.num_branches = num_ops
        self.num_cells = num_edge#args.child_num_cells
        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_keep_prob = 0 #args.lstm_keep_prob
        self.temperature = temperature
        self.tanh_constant = tanh_constant
        self.op_tanh_reduce = 2.5 #args.controller_op_tanh_reduce

        self.encoder = nn.Embedding(self.num_branches+1, self.lstm_size)

        self.lstm = nn.LSTMCell(self.lstm_size, self.lstm_size)
        self.w_soft = nn.Linear(self.lstm_size, self.num_branches, bias=False)
        b_soft = torch.zeros(1, self.num_branches)
        b_soft[:, 2:6] = 10
        self.b_soft = nn.Parameter(b_soft)
        b_soft_no_learn = np.array([0.25, 0.25, -0.25, -0.25, -0.25, -0.25, 0.25, 0.25])
        b_soft_no_learn = np.reshape(b_soft_no_learn, [1, self.num_branches])
        self.b_soft_no_learn = torch.Tensor(b_soft_no_learn).requires_grad_(False).cuda()
        # attention
        self.w_attn_1 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.w_attn_2 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.v_attn = nn.Linear(self.lstm_size, 1, bias=False)
        self.reset_param()

    def reset_param(self):
        for name, param in self.named_parameters():
            if 'b_soft' not in name:
                nn.init.uniform_(param, -0.1, 0.1)

    def forward(self):
        arc_seq_1, entropy_1, log_prob_1, c, h = self.run_sampler(use_bias=True)
        arc_seq_2, entropy_2, log_prob_2, _, _ = self.run_sampler(prev_c=c, prev_h=h)
        sample_arc = (arc_seq_1, arc_seq_2)
        sample_entropy = entropy_1 + entropy_2
        sample_log_prob = log_prob_1 + log_prob_2
        return sample_log_prob, sample_entropy, sample_arc

    def run_sampler(self, prev_c=None, prev_h=None, use_bias=False):
        if prev_c is None:
            # TODO: multi-layer LSTM
            #prev_c = [torch.zeros(1, self.lstm_size).cuda() for _ in range(self.lstm_num_layers)]
            #prev_h = [torch.zeros(1, self.lstm_size).cuda() for _ in range(self.lstm_num_layers)]
            prev_c = torch.zeros(1, self.lstm_size).cuda()
            prev_h = torch.zeros(1, self.lstm_size).cuda()

        inputs = self.encoder(torch.zeros(1).long().cuda())

        anchors = []
        anchors_w_1 = []
        arc_seq = []

        for layer_id in range(2):
            embed = inputs
            next_h, next_c = self.lstm(embed, (prev_h, prev_c))
            prev_c, prev_h = next_c, next_h
            anchors.append(torch.zeros(next_h.shape).cuda())
            anchors_w_1.append(self.w_attn_1(next_h))

        layer_id = 2
        entropy = []
        log_prob = []

        while layer_id < self.num_cells + 2:
            indices = torch.arange(0, layer_id).cuda()
            start_id = 4 * (layer_id - 2)
            prev_layers = []
            for i in range(2): # index_1, index_2
                embed = inputs
                next_h, next_c = self.lstm(embed, (prev_h, prev_c))
                prev_c, prev_h = next_c, next_h
                query = torch.stack(anchors_w_1[:layer_id], dim=1)
                query = query.view(layer_id, self.lstm_size)
                query = torch.tanh(query + self.w_attn_2(next_h))
                query = self.v_attn(query)
                logits = query.view(1, layer_id)
                if self.temperature is not None:
                    logits /= self.temperature
                if self.tanh_constant is not None:
                    logits = self.tanh_constant * torch.tanh(logits)
                prob = F.softmax(logits, dim=-1)
                index = torch.multinomial(prob, 1).long().view(1)
                arc_seq.append(index)
                arc_seq.append(0)
                curr_log_prob = F.cross_entropy(logits, index)
                log_prob.append(curr_log_prob)
                curr_ent = -torch.mean(torch.sum(torch.mul(F.log_softmax(logits, dim=-1), prob), dim=1)).detach()

                entropy.append(curr_ent)
                prev_layers.append(anchors[index])
                inputs = prev_layers[-1].view(1, -1).requires_grad_()

            for i in range(2): # op_1, op_2
                embed = inputs
                next_h, next_c = self.lstm(embed, (prev_h, prev_c))
                prev_c, prev_h = next_c, next_h
                logits = self.w_soft(next_h) + self.b_soft.requires_grad_()
                if self.temperature is not None:
                    logits /= self.temperature
                if self.tanh_constant is not None:
                    op_tanh = self.tanh_constant / self.op_tanh_reduce
                    logits = op_tanh * torch.tanh(logits)
                if use_bias:
                    logits += self.b_soft_no_learn
                prob = F.softmax(logits, dim=-1)
                op_id = torch.multinomial(prob, 1).long().view(1)
                arc_seq[2*i-3] = op_id
                curr_log_prob = F.cross_entropy(logits, op_id)
                log_prob.append(curr_log_prob)
                curr_ent = -torch.mean(torch.sum(torch.mul(F.log_softmax(logits, dim=-1), prob), dim=1)).detach()
                entropy.append(curr_ent)
                inputs = self.encoder(op_id+1)

            next_h, next_c = self.lstm(inputs, (prev_h, prev_c))
            prev_c, prev_h = next_c, next_h
            anchors.append(next_h)
            anchors_w_1.append(self.w_attn_1(next_h))
            inputs = self.encoder(torch.zeros(1).long().cuda())
            layer_id += 1

        arc_seq = torch.tensor(arc_seq)
        entropy = sum(entropy)
        log_prob = sum(log_prob)
        last_c = next_c
        last_h = next_h

        return arc_seq, entropy, log_prob, last_c, last_h
