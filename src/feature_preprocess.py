import  os, sys
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler,LabelBinarizer
from sklearn.model_selection import train_test_split
import pandas as pd
from util import utils
from util.exceptions import *
from util import daylib
from base_process import BaseProcess
from custom_dataloader import SequenceDataset,CustomTensorDataset

from debtcollector import removals
import warnings
warnings.simplefilter('always')

dl = daylib.daylib()

class featurePreprocess(BaseProcess):
    def __init__(self,_id):
        super().__init__(_id)

    def get_dataset_fn(self, dataset_fn_name):
        fns = {
            "multiseq": SequenceDataset,
            "multi": CustomTensorDataset,
        }  
        self.dataset_fn=fns.get(dataset_fn_name)
        self.dataset_fn_name = dataset_fn_name
        self._logger.info('[DONE] Get dataset fn. Function={0}'.format(self.dataset_fn_name ))
            
    def feature_label_split(self, df, target_col=None):
        """
        Split  X and y 
        Args:
            df ([type]): [description]
            target_col ([type]): [description]

        Returns:
            X: pandas.Dataframe
            y: pandas.Dataframe
        """
        if target_col is not None:
            y = df.loc[:,[target_col]]  
            X = df.drop(columns=[target_col])
        else:
            y = None
            X = df 
        return X, y

    @removals.remove
    def train_val_test_ratio_split(self, X, y, test_ratio,val_ratio):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
        self._logger.info("[DONE] Split data. Train:{0}({1}), Val:{2}({3}), Test:{4}({5})"
                          .format(len(X_train),(1-val_ratio)*(1-test_ratio),
                                  len(X_val),val_ratio*(1-test_ratio), 
                                  len(X_test),test_ratio))
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_val_test_period_split(self, dfs, train_start=None, train_end=None, valid_start=None, valid_end=None, test_start=None, test_end=None):

        trains, vals, tests = None, None, None
        try:
            if (train_start is None) or (train_start is None):
                 self._logger.info("[Skip] Split data. Dates:({0};{1}) is not appropriate".format(train_start, train_end))
            else:
                train_start, train_end = dl.intD_to_strYMD(train_start), dl.intD_to_strYMD(train_end)
                trains  = [  df.loc[train_start:train_end] for  df in  dfs ]
                self._logger.info("[DONE] Split data. Train:{0}({1}~{2})".format(len(trains[0]), train_start, train_end))
        except Exception as e:
            self._logger.warning("[Failure] Split data. Train:{0}~{1}:{2}".format(train_start, train_end,e))
            
        try:
            if (valid_start is None) or (valid_end is None):
                 self._logger.info("[Skip] Split data. Dates:({0};{1}) is not appropriate".format(valid_start, valid_end))
            else:
                valid_start, valid_end = dl.intD_to_strYMD(valid_start), dl.intD_to_strYMD(valid_end)
                vals  = [  df.loc[valid_start:valid_end] for  df in  dfs ]
                self._logger.info("[DONE] Split data. Valid:{0}({1}~{2})".format(len(vals[0]), valid_start, valid_end))
        except Exception as e:
            self._logger.warning("[Failure] Split data. Valid:{0}~{1}:{2}".format(valid_start, valid_end,e))

        try:
            if (test_start is None) or (test_end is None):
                 self._logger.info("[Skip] Split data. Dates:({0};{1}) is not appropriate".format(test_start, test_end))
            else:
                test_start, test_end = dl.intD_to_strYMD(test_start), dl.intD_to_strYMD(test_end)
                tests  = [  df.loc[test_start:test_end] for  df in  dfs ]
                self._logger.info("[DONE] Split data. Test:{0}({1}~{2})".format(len(tests[0]), test_start, test_end))
        except Exception as e:
            self._logger.warning("[Failure] Split data. Test:{0}~{1}:{2}".format( test_start, test_end,e))
        return trains, vals, tests
    
    def get_scaler(self, scaler):
        scalers = {
            "minmax": MinMaxScaler,
            "standard": StandardScaler,
            "maxabs": MaxAbsScaler,
            "robust": RobustScaler,
        }
        self._logger.info("[DONE] Get scaler:{0}".format(scaler.lower()))
        return scalers.get(scaler.lower())()

    def scalingX(self, X,  scaler=None, scaler_name=None):
        if scaler is None:
            scaler = self.get_scaler(scaler_name)
            self._logger.info(f"[DONE] New Xscaler Initilized. Name={scaler_name}")
            scaler.fit(X)            
            X_scaled = scaler.transform(X)
        else:
            X_scaled = scaler.transform(X)
        return X_scaled,scaler
        
    def get_dataloader(self,  dataset_fn, X_trains= None, y_train= None, X_vals= None,y_val= None, X_tests=None, y_test=None,  **dataset_params):
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
        batch_size = dataset_params["batch_size"]
        train_loader = None
        if X_trains is not None:
            train_features = [ torch.Tensor(X_train) for X_train in X_trains] 
            train_targets = torch.Tensor(y_train)
            
            # make dataset 
            train = self.dataset_fn(*train_features, train_targets,**dataset_params)
            
            # Convert to  loader
            train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
            self._logger.info("[DONE] Create Train Data Loader")
            
        val_loader = None          
        if X_vals is not None:
            val_features = [ torch.Tensor(X_val) for X_val in X_vals] 
            val_targets = torch.Tensor(y_val)
                        
            # make dataset 
            val = self.dataset_fn(*val_features, val_targets,**dataset_params)
            
            # Convert to  loader
            val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
            self._logger.info("[DONE] Create Valid Data Loader")
        test_loader, test_loader_one = None, None
        if X_tests is not None:
            test_features = [ torch.Tensor(X_test) for X_test in X_tests] 
            test_targets = torch.Tensor(y_test)
                        
            # make dataset 
            test = self.dataset_fn(*test_features, test_targets,**dataset_params)
            
            # Convert to  loader
            test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
            test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)
            self._logger.info("[DONE] Create Test Data Loader")
            
        return train_loader, val_loader, test_loader, test_loader_one
  
    def onehot_y(self,y):
        lb =LabelBinarizer()
        y_onehot = lb.fit_transform(y)
        idxs = y.index
        y = pd.DataFrame(y_onehot,columns=lb.classes_, index=idxs)
        return y
            
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
                
            
        
        