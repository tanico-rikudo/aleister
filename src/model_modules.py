import torch.nn as nn
import torch.nn.functional as F

class parameterParser:
    
    @staticmethod
    def sdnn(model_config):
        hparams = {}
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
    def lstm(model_config):
        hparams = {}
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
        
    def forward(self, x):
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
    

class SLSTM(nn.Module):

    def __init__(self, dimension=128):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(len(text_field.vocab), 300)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(2*dimension, 1)

    def forward(self, text, text_len):

        text_emb = self.embedding(text)

        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)

        return text_out