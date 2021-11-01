
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

import hist_data

BATCH_SIZE = 64
TEST_RATIO=0.2
VOLATILITY_VAR_DAYS = 30
VOLATILITY_DAYS = 1 
VOID_ALLOWANCE_RATIO = 1

"""
Gen data
"""
def get_load_data_proxy():
    hd = hist_data.histData()
    return hd
    
def get_trade(hd):
    trades=hd.load('BTC','trades', 20200101,20210930)
    trades.timestamp =  pd.to_datetime(trades.timestamp)
    trades.set_index("timestamp",inplace=True)
    return trade

def get_ohlcv(trades):
    ohlcv =  trades.price.resample('T', label='left', closed='left').ohlc()
    return ohlcv
    
def get_Xy(trades):
    buy_size = trades.loc[trades.loc[:,"side"]=="BUY",["size"]].resample('T', label='left', closed='left').sum().fillna(0).rename(columns={"size":"buy_size"})
    sell_size = trades.loc[trades.loc[:,"side"]=="SELL",["size"]].resample('T', label='left', closed='left').sum().fillna(0).rename(columns={"size":"sell_size"})

    mtrades = pd.concat([ohlcv,buy_size, sell_size],axis=1)
    mtrades.loc[:,["buy_size","sell_size"]] = mtrades.loc[:,["buy_size","sell_size"]].fillna(0)
    mtrades.loc[:,"close"] = mtrades.loc[:,"close"].fillna(method="ffill")
    mtrades  =  mtrades.fillna(axis=1, method='bfill')
    mtrades['size'] = mtrades.buy_size  + mtrades.sell_size

    mtrades['buy_size_ratio'] =  mtrades.buy_size /mtrades.size
    mtrades['sell_size_ratio'] =  mtrades.sell_size /mtrades.size

    ls_last_relative_target_cols =["open","high","low","close","size"]
    ls_last_relative_cols = [ "rel_"+_col for _col in  ls_last_relative_target_cols]
    convert_dict = { _before: _after for _before, _after in zip (ls_last_relative_target_cols, ls_last_relative_cols )}
    relative_values = np.log10(mtrades.loc[:,ls_last_relative_target_cols] /mtrades.last("T").loc[:,ls_last_relative_target_cols].values).rename(columns=convert_dict)

    relative_values.replace([np.inf, -np.inf],np.nan, inplace=True)
    relative_values.fillna(0,inplace=True)

    mtrades = pd.concat([mtrades, relative_values],axis=1)

    mtrades['open30Mafter']  = mtrades['open'].shift(-30)
    mtrades.loc[(mtrades['open30Mafter']  - mtrades['open']) > 0, 'movingBinary']  =  1 
    mtrades.loc[(mtrades['open30Mafter']  - mtrades['open']) == 0, 'movingBinary']  =  0
    mtrades.loc[(mtrades['open30Mafter']  - mtrades['open']) < 0, 'movingBinary']  =  -1

    mtrades.loc[:,"volatility"] = mtrades.at_time("00:00")["close"].pct_change().rolling(VOLATILITY_VAR_DAYS).std()* np.sqrt(N_VOLATILITY_DAYS)
    mtrades.loc[:,"volatility"].fillna(method="ffill",inplace=True)
    mtrades.loc[:, "price_chg_allowamce"] = mtrades.open * mtrades.volatility * 0.01 * VOID_ALLOWANCE_RATIO

    mtrades.loc[np.abs(mtrades['open30Mafter']  - mtrades['open']) <= mtrades.loc[:, "price_chg_allowamce"], "movingBinary"] = 0.0
    
    Xy = mtrades[ls_last_relative_cols+["buy_size_ratio","sell_size_ratio"]+["movingBinary"]]
    return Xy

"""
Split
"""
def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y

def train_val_test_split(df, target_col, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test

"""
Scaling
"""
def get_scaler(scaler):
    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }
    return scalers.get(scaler.lower())()

def scalingX(X,  scaler=None):
    if scaler is None:
        scaler = get_scaler('minmax')
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.fit_transform(X)
    return X_scaled

"""
Tensor
"""
def convert_tensor(X_train, y_train, X_val,y_val, X_test, y_test):
    train_features = torch.Tensor(X_train)
    train_targets = torch.Tensor(y_train)
    val_features = torch.Tensor(X_val)
    val_targets = torch.Tensor(y_val)
    test_features = torch.Tensor(X_test)
    test_targets = torch.Tensor(y_test)

    train = TensorDataset(train_features, train_targets)
    val = TensorDataset(val_features, val_targets)
    test = TensorDataset(test_features, test_targets)

    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)
    
"""
Master
"""
def exec():
    hd = get_load_data_proxy()
    trades = get_trade(hd)
    ohlcv = get_ohlcv(trades)
    Xy = build_Xy(trades)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(Xy, 'movingBinary', TEST_RATIO)
    X_train, X_val, X_test, y_train, y_val, y_test  = X_train.values, X_val.values, X_test.values, y_train.values, y_val.values, y_test.values
    ls_cols = X_train.columns

    scaler  = get_scaler("")
    X_val_arr = (X_val)
    X_test_arr = scaler.transform(X_test)
    

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
