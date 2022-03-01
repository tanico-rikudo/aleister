import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class Attentive_Pooling(nn.Module):
    def __init__(self, hidden_dim):
        """

        :param hidden_dim:
        """
        super(Attentive_Pooling, self).__init__()
        self.w_1 = nn.Linear(hidden_dim, hidden_dim)
        self.w_2 = nn.Linear(hidden_dim, hidden_dim)
        self.u = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, memory, query=None, mask=None):
        '''

        :param query:   (node, hidden)
        :param memory: (node, hidden)
        :param mask:
        :return:
        '''
        if query is None:
            h = torch.tanh(self.w_1(memory))  # node, hidden
        else:
            h = torch.tanh(self.w_1(memory) + self.w_2(query))
        score = torch.squeeze(self.u(h), -1)  # node,
        if mask is not None:
            score = score.masked_fill(mask.eq(0), -1e9)
        alpha = F.softmax(score, -1)  # node,
        s = torch.sum(torch.unsqueeze(alpha, -1) * memory, -2)
        return s


class RGCN(nn.Module):
    def __init__(self, in_features, out_features, relation_num):
        """

        :param in_features:
        :param out_features:
        :param relation_num:
        """
        super(RGCN, self).__init__()
        self.relation_num = relation_num
        self.linears = [nn.Linear(in_features, out_features) for i in range(relation_num) ]
        self.activation = nn.Tanh()

    def gcn(self, relation, input, adj):
        """

        :param relation:
        :param input:
        :param adj:
        :return:
        """
        support = self.linears[relation](input)
        output = torch.sparse.mm(adj, support)
        return output

    def forward(self, input, adjs):
        '''

        :param input:   (node, hidden)
        :param adjs:    (node, node)
        :return:
        '''
        transform = []
        for r in range(self.relation_num):
            transform.append(self.gcn(r, input, adjs[r]))
        # (node, relation, hidden) -> (node, hidden)
        # Note: Activation on Relation-axis
        return self.activation(torch.sum(torch.stack(transform, 1), 1))


class SLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_dim, relation_num, dropout):
        """

        :param input_size:
        :param hidden_dim:
        :param relation_num:
        :param dropout:
        """
        super(SLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = torch.nn.Dropout(dropout)
        self.Wh = nn.Linear(hidden_dim, hidden_dim * 5, bias=False)
        self.Wn = nn.Linear(hidden_dim, hidden_dim * 5, bias=False)
        self.Wt = nn.Linear(hidden_dim, hidden_dim * 5, bias=False)
        self.U = nn.Linear(input_size, hidden_dim * 5, bias=False)
        self.V = nn.Linear(hidden_dim, hidden_dim * 5)
        self.rgcn = RGCN(hidden_dim, hidden_dim, relation_num)

    def forward(self, x, h, c, g, h_t, adjs):
        '''

        :param x:   (node, emb)
            embedding of the node, news and initial node embedding
        :param h:   (node, hidden)
            hidden state from last layer
        :param c:   candidate from last layer
        :param g:   (hidden)
            hidden state of the global node
        :param h_t:   (node, hidden)
            hidden state from last time
        :param adj:   (node, node)
            if use RGCN, there should be multiple gcns, each one for a relation
        :return:
        '''
        # adjs = [adj]
        hn = self.rgcn(h, adjs)
        gates = self.Wh(self.dropout(h)) + self.U(self.dropout(x)) + self.Wn(self.dropout(hn)) + self.Wt(
            self.dropout(h_t)) + torch.unsqueeze(self.V(g), 0)
        i, f, o, u, t = torch.split(gates, self.hidden_dim, dim=-1)
        new_c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(u) + torch.sigmoid(t) * h_t
        new_h = torch.sigmoid(o) * torch.tanh(new_c)
        return new_h, new_c


class GLSTMCell(nn.Module):
    def __init__(self, hidden_dim, attn_pooling):
        """

        :param hidden_dim:
        :param attn_pooling:
        """
        super(GLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.w = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U = nn.Linear(hidden_dim, hidden_dim * 2)
        self.u = nn.Linear(hidden_dim, hidden_dim)
        self.attn_pooling = attn_pooling

    def forward(self, g, c_g, h, c):
        """

        :param g:
        :param c_g:
        :param h:
        :param c:
        :return:
        """
        ''' assume dim=1 is word'''
        # this can use attentive pooling
        # h_avg = torch.mean(h, 1)
        h_avg = self.attn_pooling(h)
        f, o = torch.split(torch.sigmoid(self.W(g) + self.U(h_avg)), self.hidden_dim, dim=-1)
        f_w = torch.sigmoid(torch.unsqueeze(self.w(g), -2) + self.u(h))
        f_w = F.softmax(f_w, -2)
        new_c = f * c_g + torch.sum(c * f_w, -2)
        new_g = o * torch.tanh(new_c)
        return new_g, new_c


class CGM(nn.Module):
    def __init__(self, hidden_dim, vol_input_size, price_input_size, seq_dropout_rate, gbl_dropout_rate,
                 last_dropout_rate, relation_num, output_dim, num_layers):
        super(CGM, self).__init__()
        self.hidden_dim = hidden_dim

        self.feature_weight_price = nn.Linear(price_input_size, hidden_dim)
        self.feature_weight_volume = nn.Linear(vol_input_size, hidden_dim)
        self.feature_combine = nn.Linear(hidden_dim * 4, hidden_dim)

        self.cca_price = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
                                       nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.ReLU(),
                                       nn.Linear(hidden_dim * 2, hidden_dim))
        self.cca_volume = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
                                        nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.ReLU(),
                                        nn.Linear(hidden_dim * 2, hidden_dim))

        self.attn_pooling = Attentive_Pooling(hidden_dim)
        self.s_cell = SLSTMCell(hidden_dim, hidden_dim, relation_num, seq_dropout_rate)
        self.g_cell = GLSTMCell(hidden_dim, self.attn_pooling, gbl_dropout_rate)
        self.w_out = nn.Linear(hidden_dim, output_dim)
        self.num_layers = num_layers
        self.dropout = torch.nn.Dropout(last_dropout_rate)

    def encode(self, span_nodes, node_feature, adj):
        """

        :param span_nodes: (time, node)
        :param text_nodes: (node)
        :param node_text:   (node, seq)
        :param text_length:
        :param text_mask:  (node, seq)
        :param node_feature:   (time, node, feature_size)
        :param adj:
        :return:
        """
        # TODO: node_feature order and contents
        node_emb = self.node_emb(span_nodes)  # idx of node
        x_price = self.feature_weight_price(node_feature[:, :, :6])
        x_volume = self.feature_weight_volume(node_feature[:, :, 6:])

        ### price graph ###
        last_h_time_price = torch.squeeze(x_price[0], 0)  # node feature 
        last_c_time_price = torch.squeeze(x_price[0], 0)  # node feature 
        
        last_g_time_price = self.attn_pooling(last_h_time_price, node_emb)
        last_c_g_time_price = self.attn_pooling(last_c_time_price, node_emb)
        # h_states = []
        
        time = node_feature.size(0)
        for t in range(time):
            # init
            last_h_layer_price = last_h_time_price
            last_c_layer_price = last_c_time_price
            # information integration 
            last_g_layer_price, last_c_g_layer_price = self.g_cell.init_forward(last_g_time_price, last_c_g_time_price,
                                                                                last_h_layer_price, last_c_layer_price,
                                                                                node_emb)
            for l in range(self.num_layers):
                # x, h, c, g, h_t, adj
                last_h_layer_price, last_c_layer_price = self.s_cell(torch.squeeze(x_price[t], 0), last_h_layer_price,
                                                                     last_c_layer_price, last_g_layer_price,
                                                                     last_h_time_price, adj)
                # g, c_g, t_g, t_c, h, c
                last_g_layer_price, last_c_g_layer_price = self.g_cell(last_g_layer_price, last_c_g_layer_price,
                                                                       last_g_time_price, last_c_g_time_price,
                                                                       last_h_layer_price, last_c_layer_price, node_emb)
            last_h_time_price, last_c_time_price = last_h_layer_price, last_c_layer_price
            last_g_time_price, last_c_g_time_price = last_g_layer_price, last_c_g_layer_price

        ### volume graph ###
        last_h_time_volume = torch.squeeze(x_volume[0], 0)  # node feature 
        last_c_time_volume = torch.squeeze(x_volume[0], 0)  # node feature 
        last_g_time_volume = self.attn_pooling(last_h_time_volume, node_emb)
        last_c_g_time_volume = self.attn_pooling(last_c_time_volume, node_emb)
        # h_states = []
        
        time = node_feature.size(0)
        for t in range(time):
            # init
            last_h_layer_volume = last_h_time_volume
            last_c_layer_volume = last_c_time_volume
            # information integration 
            last_g_layer_volume, last_c_g_layer_volume = self.g_cell.init_forward(last_g_time_volume,
                                                                                  last_c_g_time_volume,
                                                                                  last_h_layer_volume,
                                                                                  last_c_layer_volume, node_emb)
            for l in range(self.num_layers):
                # x, h, c, g, h_t, adj
                last_h_layer_volume, last_c_layer_volume = self.s_cell(torch.squeeze(x_volume[t], 0),
                                                                       last_h_layer_volume, last_c_layer_volume,
                                                                       last_g_layer_volume, last_h_time_volume, adj)
                # g, c_g, t_g, t_c, h, c
                last_g_layer_volume, last_c_g_layer_volume = self.g_cell(last_g_layer_volume, last_c_g_layer_volume,
                                                                         last_g_time_volume, last_c_g_time_volume,
                                                                         last_h_layer_volume, last_c_layer_volume,
                                                                         node_emb)
            last_h_time_volume, last_c_time_volume = last_h_layer_volume, last_c_layer_volume
            last_g_time_volume, last_c_g_time_volume = last_g_layer_volume, last_c_g_layer_volume

        ### CCA ###
        cca_price, cca_volume = self.cca_price(last_h_time_price), self.cca_volume(last_h_time_volume)
        
        
        last_h_layer, last_c_layer, last_g_layer, last_c_g_layer = last_h_layer_volume, last_c_layer_volume, last_g_layer_volume, last_c_g_layer_volume
        for l in range(self.num_layers):
            last_h_layer, last_c_layer = self.text_s_cell(node_vector, last_h_layer, last_c_layer, last_g_layer, adj)
            # g, c_g, h, c
            last_g_layer, last_c_g_layer = self.text_g_cell(last_g_layer, last_c_g_layer, last_h_layer, last_c_layer)

        return last_h_layer, cca_price, cca_volume

    def forward(self, span_nodes, node_text, text_mask, node_feature, adj):
        '''
        :param batch:
            nodes: (time, graph_node)
            node_text: (time, node, seq)
            adjs: (time, node, node)
        :param use_cuda:
        :return:    (node, label)
        '''
        text_lengths = text_mask.sum(-1).int()
        assert text_lengths.max() <= node_text.size(-1) and text_lengths.min() > 0, (text_lengths, node_text.size())
        last_h, cca_price, cca_volume = self.encode(span_nodes, node_feature, adj)
        # the index 0 here is the first time step, which is because that this is the initialization
        return self.w_out(self.dropout(last_h)), cca_price, cca_volume
