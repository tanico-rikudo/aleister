import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split

class featurePreprocess:
    def __init__(self,logger):
        self._logger  = logger
    
    def feature_label_split(self, df, target_col):
        y = df[[target_col]]
        X = df.drop(columns=[target_col])
        return X, y

    def train_val_test_split(self, df, target_col, test_ratio,val_ratio):
        val_ratio = test_ratio / (1 - test_ratio)
        X, y = self.feature_label_split(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
        self._logger.info("[DONE] Split data. Train:{0}({1}), Val:{2}({3}), Test:{4}({5})"
                          .format(len(X_train),(1-val_ratio),len(X_train),val_ratio, len(X_train),test_ratio))
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_scaler(self, scaler):
        scalers = {
            "minmax": MinMaxScaler,
            "standard": StandardScaler,
            "maxabs": MaxAbsScaler,
            "robust": RobustScaler,
        }
        self._logger.info("[DONE] Get scaler:{0}".format(scaler.lower()))
        return scalers.get(scaler.lower())()

    def scalingX(self, X,  scaler=None):
        if scaler is None:
            scaler = self.get_scaler('minmax')
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)
        return X_scaled,scaler
    
    def get_dataloader(self, X_train, y_train, X_val,y_val, X_test=None, y_test=None, batch_size=64):
        
        train_features = torch.Tensor(X_train)
        train_targets = torch.Tensor(y_train)
        train = TensorDataset(train_features, train_targets)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
        
        
        val_features = torch.Tensor(X_val)
        val_targets = torch.Tensor(y_val)
        val = TensorDataset(val_features, val_targets)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
        
        test_loader, test_loader_one = None, None
        if X_test is not None:
            test_features = torch.Tensor(X_test)
            test_targets = torch.Tensor(y_test)
            test = TensorDataset(test_features, test_targets)
            test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
            test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)
        self._logger.info("[DONE] Data Loader")
        return train_loader, val_loader, test_loader, test_loader_one

    def convert_dataset(self, Xy, ans_col, test_ratio=0.4, valid_ratio=0.5):
        X_train, X_val, X_test, y_train, y_val, y_test = self.train_val_test_split(Xy, ans_col, test_ratio, valid_ratio)
        X_train, X_val, X_test, y_train, y_val, y_test  = X_train.values, X_val.values, X_test.values, y_train.values, y_val.values, y_test.values
        return X_train, X_val, X_test, y_train, y_val, y_test