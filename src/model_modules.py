import torch.nn as nn
import torch.nn.functional as F

class parameterParser:
    
    @staticmethod
    def sdnn(model_config):
        hparams = {}
        hparams["structure_params"] = ["hidden_dim","layer_dim","out_dim","l2_drop_rate"]
        hparams["n_epoch"] = model_config.getint("N_EPOCH")
        hparams["batch_size"] = model_config.getint("BATCHSIZE")
        hparams["hidden_dim"] = model_config.getint("HIDDEN_DIM")
        hparams["layer_dim"] = model_config.getint("LAYER_DIM")
        hparams["out_dim"] = model_config.getint("OUT_DIM")
        hparams["l2_drop_rate"] = model_config.getfloat("L2_DROP_RATE")
        hparams["weight_decay"] = model_config.getfloat("WEIGHT_DECAY")
        hparams["lr"] = model_config.getfloat("LR")
        hparams["optimizer"] = model_config.get("OPTIMIZER")
        hparams["loss_fn"] = model_config.get("LOSSFN")
        return hparams
    
    @staticmethod
    def slstm(model_config):
        hparams = {}
        hparams["structure_params"] = ["hidden_dim","layer_dim","out_dim","l2_drop_rate"]
        hparams["n_epoch"] = model_config.getint("N_EPOCH")
        hparams["batch_size"] = model_config.getint("BATCHSIZE")
        hparams["hidden_dim"] = model_config.getint("HIDDEN_DIM")
        hparams["layer_dim"] = model_config.getint("LAYER_DIM")
        hparams["out_dim"] = model_config.getint("OUT_DIM")
        hparams["l2_drop_rate"] = model_config.getfloat("L2_DROP_RATE")
        hparams["weight_decay"] = model_config.getfloat("WEIGHT_DECAY")
        hparams["lr"] = model_config.getfloat("LR")
        hparams["optimizer"] = model_config.get("OPTIMIZER")
        hparams["loss_fn"] = model_config.get("LOSSFN")
        return hparams
        
    
class SimpleDnn(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, l2_drop_rate):
        super(SimpleDnn, self).__init__()
        
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, layer_dim)
        self.layer_3 = nn.Linear(layer_dim, layer_dim//2)
        self.layer_out = nn.Linear(layer_dim//2, output_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=l2_drop_rate)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)
        self.batchnorm2 = nn.BatchNorm1d(layer_dim)
        # self.batchnorm3 = nn.BatchNorm1d(64)
        
    def forward(self, xs):
        x = xs[0]
        x = self.layer_1(x)
        # x = self.batchnorm1(x)
        # x = self.relu(x)
        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_3(x)
        # x = self.batchnorm3(x)
        # x = self.relu(x)
        x = self.dropout(x)    
        
        x = self.layer_out(x)
        x = F.softmax(x, dim=1)
        
        return x
    

class SimpleLSTM(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super(SimpleLSTM, self).__init__()
        
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, xs):
        x = xs[0]
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_dim)
        
        out = self.fc(h_out)
        
        return out