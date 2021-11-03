
import os,sys
os.environ['BASE_DIR'] =  '/Users/macico/Dropbox/btc'
os.environ['KULOKO_DIR'] = os.path.join(os.environ['BASE_DIR'], "kuloko")
os.environ['COMMON_DIR'] = os.path.join(os.environ['BASE_DIR'], "geco_commons")
os.environ['MONGO_DIR'] = os.path.join(os.environ['COMMON_DIR'] ,"mongodb")
os.environ['LOGDIR'] = os.path.join(os.environ['KULOKO_DIR'], "log")
sys.path.append(os.path.join(os.environ['KULOKO_DIR'],"items" ))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

torch.manual_seed(71) 
    
def do_train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{device}" " is available.")

    input_dim = len(ls_cols)
    output_dim = 1
    hidden_dim = 32 # 256
    layer_dim = 64 #512
    batch_size = 64
    dropout = 0.2
    n_epochs = 10
    learning_rate = 1e-3
    weight_decay = 1e-6
    model_name  = 'sdnn'

    model_params = {'input_dim': input_dim,
                    'hidden_dim' : hidden_dim,
                    'layer_dim' : layer_dim,
                    'output_dim' : output_dim}

    model = get_model(model_name, model_params)


    # criterion
    loss_fn = nn.BCEWithLogitsLoss()#(reduction="mean")
    # loss_fn=nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    le = LearningEvaluator(model=model, loss_fn=loss_fn, optimizer=optimizer, name=model_name)
    le.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
    le.plot_losses()

    predictions, values = opt.evaluate(
        test_loader_one,
        batch_size=1,
        n_features=input_dim
    )
