import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SimpleLSTM(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, window_size):
        super(SimpleLSTM, self).__init__()

        self.output_dim = output_dim
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size

        # batch * window * dim
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.window_size, -1)  # batch * window * dim
        h_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim), requires_grad=True)
        c_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim), requires_grad=True)

        # Propagate input through LSTM
        lstm_out, (h_out, c_out) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(self.num_layers, batch_size, self.hidden_dim)[-1]  # num_layer * batch * dim

        out = self.fc(h_out)
        out = F.softmax(out, dim=1)

        return out


