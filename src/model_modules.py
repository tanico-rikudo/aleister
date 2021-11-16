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
        x = F.softmax(x, dim=0)
        
        return x
    
