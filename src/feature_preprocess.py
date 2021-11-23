import  os, sys
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler,LabelBinarizer
from sklearn.model_selection import train_test_split
import pandas as pd
from util import utils
from util.exceptions import *
from util import daylib
from base_process import BaseProcess

dl = daylib.daylib()

class featurePreprocess(BaseProcess):
    def __init__(self,_id):
        super().__init__(_id)
            
    def feature_label_split(self, df, target_col):
        y = df.loc[:,[target_col]]
        X = df.drop(columns=[target_col])
        return X, y

    def train_val_test_ratio_split(self, X, y, test_ratio,val_ratio):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
        self._logger.info("[DONE] Split data. Train:{0}({1}), Val:{2}({3}), Test:{4}({5})"
                          .format(len(X_train),(1-val_ratio)*(1-test_ratio),
                                  len(X_val),val_ratio*(1-test_ratio), 
                                  len(X_test),test_ratio))
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_val_test_period_split(self, X, y, train_start=None, train_end=None, valid_start=None, valid_end=None, test_start=None, test_end=None):
        try:
            train_start, train_end = dl.intD_to_strYMD(train_start), dl.intD_to_strYMD(train_end)
            X_train, y_train = X.loc[train_start:train_end], y.loc[train_start:train_end]
            self._logger.info("[DONE] Split data. Train:{0}({1}~{2})".format(len(X_train), train_start, train_end))
        except Exception as e:
            self._logger.warning("[Failure] Split data. Train:{0}~{1}})".format(train_start, train_end))
        try:
            valid_start, valid_end = dl.intD_to_strYMD(valid_start), dl.intD_to_strYMD(valid_end)
            X_val, y_val = X.loc[valid_start:valid_end], y.loc[valid_start:valid_end]
            self._logger.info("[DONE] Split data. Valid:{0}({1}~{2})".format(len(X_val), valid_start, valid_end))
        except Exception as e:
            self._logger.warning("[Failure] Split data. Valid:{0}~{1}".format(valid_start, valid_end))

        try:
            test_start, test_end = dl.intD_to_strYMD(test_start), dl.intD_to_strYMD(test_end)
            X_test, y_test = X.loc[test_start:test_end], y.loc[test_start:test_end]
            self._logger.info("[DONE] Split data. Test:{0}({1}~{2})".format(len(X_test), test_start, test_end))
        except Exception as e:
            self._logger.warning("[Failure] Split data. Test:{0}~{1}".format( test_start, test_end))
        
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
    
    def get_dataloader(self, X_train= None, y_train= None, X_val= None,y_val= None, X_test=None, y_test=None, batch_size=64):
        train_loader = None
        if X_train is not None:
            train_features = torch.Tensor(X_train)
            train_targets = torch.Tensor(y_train)
            
            # make dataset 
            train = TensorDataset(train_features, train_targets)
            
            # Convert to  loader
            train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
            
        val_loader = None          
        if X_val is not None:
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
        self._logger.info("[DONE] Create Data Loader")
        return train_loader, val_loader, test_loader, test_loader_one
    
    def get_multi_dataloader(self, X_trains= None, y_train= None, X_vals= None,y_val= None, X_tests=None, y_test=None, batch_size=64):
        """
        Create Dataloader feeding multi-input
        Args:
            X_trains (list, optional): ndarray list. Defaults to None.
            y_train (ndarray, optional): answer. Defaults to None.
            X_vals (list, optional): ndarray list. Defaults to None.
            y_val (ndarray, optional): answer. Defaults to None.
            X_tests (ndarray, optional): ndarray list. Defaults to None.
            y_test (ndarray, optional): answer. Defaults to None.
            batch_size (int, optional): batch_size. Defaults to 64.

        Returns:
            tuple : dataloaders
        """
        train_loader = None
        if X_train is not None:
            train_features = [ torch.Tensor(X_train) for X_train in X_trains] 
            train_targets = torch.Tensor(y_train)
            
            # make dataset 
            train = TensorDataset(*train_features, train_targets)
            
            # Convert to  loader
            train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
            
        val_loader = None          
        if X_val is not None:
            val_features = [ torch.Tensor(X_val) for X_val in X_vals] 
            val_targets = torch.Tensor(y_val)
            val = TensorDataset(*val_features, val_targets)
            val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
        
        test_loader, test_loader_one = None, None
        if X_test is not None:
            test_features = [ torch.Tensor(X_test) for X_test in X_tests] 
            test_targets = torch.Tensor(y_test)
            test = TensorDataset(*test_features, test_targets)
            test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
            test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)
        self._logger.info("[DONE] Create Data Loader")
        return train_loader, val_loader, test_loader, test_loader_one

    def convert_dataset(self, Xy, ans_col, split_rule, test_ratio=0.4, valid_ratio=0.5, 
                        train_start=None, train_end=None, valid_start=None, valid_end=None, test_start=None, test_end=None):
        X, y = self.feature_label_split(df=Xy, target_col=ans_col)

        lb =LabelBinarizer()
        y_onehot = lb.fit_transform(y)
        idxs = y.index
        y = pd.DataFrame(y_onehot,columns=lb.classes_, index=idxs)
        if split_rule == 'ratio':
            X_train, X_val, X_test, y_train, y_val, y_test = self.train_val_test_ratio_split(X, y, test_ratio, valid_ratio)
        elif split_rule == 'period':
            X_train, X_val, X_test, y_train, y_val, y_test = \
                self.train_val_test_period_split(X, y, train_start, train_end, valid_start, valid_end, test_start, test_end)
        else:
            logging.warning("[Failure] No split rule.")
        X_train, X_val, X_test, y_train, y_val, y_test  = X_train.values, X_val.values, X_test.values, y_train.values, y_val.values, y_test.values
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_numpy_datas(self,**kwargs):
        for key, obj in kwargs.items():
            save_path = os.path.join(self.save_dir, "{0}.jbl".format(key) )
            try:
                utils.saveJbl(obj,save_path)
                self._logger.info("[DONE] Save Prepro data. Key={0}, path={1}".format(key, save_path))
            except Exception as e:
                self._logger.warning("[Failure] Save Prepro data. Key={0}, path={1}".format(key, save_path),exc_info=True)
                
    def load_numpy_datas(self,args):
        objs = {}
        for key in args:
            load_path = os.path.join(self.save_dir, "{0}.jbl".format(key) )
            try:
                obj = utils.loadJbl(load_path)
                self._logger.info("[DONE] Load Prepro data. Key={0}, path={1}".format(key, load_path))
                objs[key] = obj
            except Exception as e:
                self._logger.warning("[Failure] Load Prepro data. Key={0}, path={1}".format(key, load_path))
        return (objs[key] for key in args)
                
            
        
        