from datetime import datetime as dt
from datetime import timedelta as ddlta
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class RGCN(nn.Module):
    def __init__(self, in_features, out_features, relation_num):
        """
        RGCN module
        :param in_features:
        :param out_features:
        :param relation_num:
        """
        super(RGCN, self).__init__()
        self.relation_num = relation_num
        # Create (# relation ) layers
        self.linears = [nn.Linear(in_features, out_features) for i in range(relation_num)]
        self.activation = nn.Tanh()

    def gcn(self, relation, input, adj):
        """

        :param relation:
        :param input:
        :param adj:
        :return:
        """
        support = self.linears[relation](input)  # support = liner(input)
        output = torch.sparse.mm(adj, support)  # adj * support
        return output

    def forward(self, input, adjs):
        '''
        :param input:   (node, hidden)
        :param adjs:    (relation, node)
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
        2nd Laler using
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
        # attention (pre)
        h = torch.reshape(h, (-1, h.shape[-1]))
        hn = self.rgcn(h, adjs)

        # LSTM gates calculation
        gates = self.Wh(self.dropout(h)) + \
                self.U(self.dropout(x)) + \
                self.Wn(self.dropout(hn)) + \
                self.Wt(self.dropout(h_t)) + \
                torch.unsqueeze(self.V(g), 0)

        i, f, o, u, t = torch.split(gates, self.hidden_dim, dim=-1)
        new_c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(u) + torch.sigmoid(t) * h_t
        new_h = torch.sigmoid(o) * torch.tanh(new_c)

        # Note: Sometimes shape is changed ...... reshape forcefully
        new_c = torch.reshape(new_c, (1, -1, new_c.shape[-1]))
        new_h = torch.reshape(new_h, (1, -1, new_h.shape[-1]))

        return new_h, new_c


class GLSTMCell(nn.Module):
    def __init__(self, hidden_dim, dropout):
        """

        :param hidden_dim:
        """
        super(GLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = torch.nn.Dropout(dropout)
        self.W = nn.Linear(hidden_dim, hidden_dim * 5, bias=False)
        self.w = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U = nn.Linear(hidden_dim, hidden_dim * 5)
        self.u = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, g, c_g, t_g, t_c, h, c):
        """

        :param g:
        :param c_g:
        :param h:
        :param c:
        :return:
        """

        # attention (pre)
        h_avg = torch.mean(h, 1)  # Note: Originaal has attentio.  No attention pooling because of no words

        # LSTM gates
        gates = torch.sigmoid(self.W(g) + self.U(h_avg))

        i, f, o, u, _ = torch.split(gates, self.hidden_dim, dim=-1)
        f_w = torch.sigmoid(torch.unsqueeze(self.w(g), -2) + self.u(h))
        f_w = F.softmax(f_w, -2)
        new_c = f * c_g + torch.sum(c * f_w, -2)
        new_g = o * torch.tanh(new_c)

        return new_g, new_c

    def init_forward(self, t_g, t_c, h):
        # h_avg = torch.mean(h, 1)
        h_avg = h
        # the gates are calculated according to h
        gates = self.dropout(self.W(t_g) + self.U(h_avg))
        i, f, o, u, _ = torch.split(gates, self.hidden_dim, dim=-1)
        new_c = torch.sigmoid(f) * t_c + torch.sigmoid(i) * torch.tanh(u)
        new_g = o * torch.tanh(new_c)
        return new_g, new_c


class CGM(nn.Module):
    def __init__(self, hidden_dim, vol_input_size, price_input_size, seq_dropout_rate, gbl_dropout_rate,
                 last_dropout_rate, relation_num, output_dim, num_layers, input_dim):
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

        #
        self.s_cell = SLSTMCell(
            input_size=hidden_dim,
            hidden_dim=hidden_dim,
            relation_num=relation_num,
            dropout=seq_dropout_rate)

        self.g_cell = GLSTMCell(hidden_dim, gbl_dropout_rate)
        self.w_out = nn.Linear(hidden_dim, output_dim)
        self.num_layers = num_layers
        self.dropout = torch.nn.Dropout(last_dropout_rate)

    def encode(self, price_data, volume_data):
        """

        :param price_data:
        :param volume_data:
        :param adj:
        :return:
        """

        x_price = self.feature_weight_price(price_data[:, :, :self.price_input_size])
        x_volume = self.feature_weight_volume(volume_data[:, :, :self.vol_input_size])
        time_len = price_data.shape[0]

        price_graph, price_graph_np = self.build_graph(x_price.detach().numpy(), 5)
        volume_graph, volume_graph_np = self.build_graph(x_volume.detach().numpy(), 5)

        volume_graph_np = torch.from_numpy(np.array(volume_graph_np)).float()
        price_graph_np = torch.from_numpy(np.array(price_graph_np)).float()

        last_h_time_price = torch.squeeze(x_price[0], 0)  # node feature
        last_c_time_price = torch.squeeze(x_price[0], 0)  # node feature

        last_h_time_volume = torch.squeeze(x_volume[0], 0)  # node feature
        last_c_time_volume = torch.squeeze(x_volume[0], 0)  # node feature

        last_g_time_price = last_h_time_price
        last_c_g_time_price = last_c_time_price

        last_g_time_volume = last_h_time_volume
        last_c_g_time_volume = last_c_time_volume

        for t in range(time_len):
            # init
            last_h_layer_price = last_h_time_price
            last_c_layer_price = last_c_time_price
            # information integration
            # Each input: Comps * hidden size
            last_g_layer_price, last_c_g_layer_price = self.g_cell.init_forward(last_g_time_price,
                                                                                last_c_g_time_price,
                                                                                last_h_layer_price)
            for l in range(self.num_layers):
                # x, h, c, g, h_t, adj
                last_h_layer_price, last_c_layer_price = self.s_cell(
                    torch.squeeze(x_price[t], 0),
                    last_h_layer_price,
                    last_c_layer_price,
                    last_g_layer_price,
                    last_h_time_price,
                    price_graph_np)
                # g, c_g, t_g, t_c, h, c
                last_g_layer_price, last_c_g_layer_price = self.g_cell(
                    last_g_layer_price,
                    last_c_g_layer_price,
                    last_g_time_price,
                    last_c_g_time_price,
                    last_h_layer_price,
                    last_c_layer_price)

            last_h_time_price, last_c_time_price = last_h_layer_price, last_c_layer_price
            last_g_time_price, last_c_g_time_price = last_g_layer_price, last_c_g_layer_price

        for t in range(time_len):
            # init
            last_h_layer_volume = last_h_time_volume
            last_c_layer_volume = last_c_time_volume
            # information integration
            last_g_layer_volume, last_c_g_layer_volume = self.g_cell.init_forward(
                last_g_time_volume,
                last_c_g_time_volume,
                last_h_layer_volume,
            )
            for l in range(self.num_layers):
                # x, h, c, g, h_t, adj
                last_h_layer_volume, last_c_layer_volume = self.s_cell(torch.squeeze(x_volume[t], 0),
                                                                       last_h_layer_volume,
                                                                       last_c_layer_volume,
                                                                       last_g_layer_volume,
                                                                       last_h_time_volume,
                                                                       volume_graph_np)
                # g, c_g, t_g, t_c, h, c
                last_g_layer_volume, last_c_g_layer_volume = self.g_cell(last_g_layer_volume,
                                                                         last_c_g_layer_volume,
                                                                         last_g_time_volume,
                                                                         last_c_g_time_volume,
                                                                         last_h_layer_volume,
                                                                         last_c_layer_volume,
                                                                         )

            last_h_time_volume, last_c_time_volume = last_h_layer_volume, last_c_layer_volume
            last_g_time_volume, last_c_g_time_volume = last_g_layer_volume, last_c_g_layer_volume

        ### CCA ###
        cca_price, cca_volume = self.cca_price(last_h_time_price), self.cca_volume(last_h_time_volume)
        last_h_layer, last_c_layer, last_g_layer, last_c_g_layer = last_h_layer_volume, last_c_layer_volume, last_g_layer_volume, last_c_g_layer_volume

        return last_h_layer, cca_price, cca_volume

    def build_graph(data, n_index, n_feature):
        graph = {}
        np_graph = []
        for _feature_key in range(n_feature):
            graph[_feature_key] = {}
            layer_1 = []
            for _idx_key_1 in range(len(n_index)):
                graph[_feature_key][_idx_key_1] = {}
                layer_2 = []
                for _idx_key_2 in range(len(n_index)):
                    ir, p = scipy.stats.pearsonr(data[_idx_key_1, :, _feature_key], data[_idx_key_2, :, _feature_key])
                    graph[_feature_key][_idx_key_1][_idx_key_2] = ir
                    layer_2.append(ir)
                layer_1.append(layer_2)
            np_graph.append(layer_1)
        return graph, np_graph

    def forward(self, price_data, volume_data):
        '''
        :param span_nodes:
        :param batch:
            nodes: (time, graph_node)
            node_feature: (time, node, seq)
            adjs: (time, node, node)
        :param use_cuda:
        :return:    (node, label)
        '''
        last_h, cca_price, cca_volume = self.encode(price_data, volume_data)
        # the index 0 here is the first time step, which is because that this is the initialization
        return self.w_out(self.dropout(last_h)), cca_price, cca_volume
