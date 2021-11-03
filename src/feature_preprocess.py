import data_gen 


BATCH_SIZE = 64
TEST_RATIO=0.2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

class featurePreprocess:
    
    def __init__():
        self.
    
    @staticmethod
    def feature_label_split(df, target_col):
        y = df[[target_col]]
        X = df.drop(columns=[target_col])
        return X, y

    @staticmethod
    def train_val_test_split(df, target_col, test_ratio,val_ratio):
        val_ratio = test_ratio / (1 - test_ratio)
        X, y = feature_label_split(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
        return X_train, X_val, X_test, y_train, y_val, y_test
    
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
            X_scaled = scaler.transform(X)
        return X_scaled,scaler
    
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
        
    def exec():
        dg =  data_gen()
        dg.get_load_data_proxy()
        trades = dg.get_trade()
        ohlcv = dg.get_ohlcv(trades)
        Xy = build_Xy(trades)

        X_train, X_val, X_test, y_train, y_val, y_test = self.train_val_test_split(Xy, 'movingBinary', TEST_RATIO, VALID_RATIO)
        X_train, X_val, X_test, y_train, y_val, y_test  = X_train.values, X_val.values, X_test.values, y_train.values, y_val.values, y_test.values
        ls_cols = X_train.columns

        scaler  = self.get_scaler("minmax")
        X_train,x_scaler = self.scalingX(X_train)
        X_val,_ = self.scalingX(X_val,x_scaler)
        X_test,_ = self.scalingX(X_test,x_scaler)
 
        datsets={"train":(X_train,y_train),
            "valid":(X_val,y_val)}       
        if TEST_RATIO  > 0:
            datsets["test"]=(X_test,y_test)
            
        return datsets
