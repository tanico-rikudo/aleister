import torch
from torch.utils.data import Dataset

import numpy  as np

class SequenceDataset(Dataset):
    def __init__(self, *tensors, **params):
        self.X_tensors = tensors[:-1]
        self.y_tensor = tensors[-1]
        self.window_size = params["window_size"]
        self.batch_size = params["batch_size"]
        self.Xs = [ X_tensor.numpy() for X_tensor in self.X_tensors]
        self.y = self.y_tensor.numpy()

    def __len__(self):
        return self.Xs[0].shape[0]

    def __getitem__(self, idx): 
        if idx >= (self.window_size - 1):
            idx_start = idx - self.window_size + 1
            Xs = []
            for X in self.Xs:
                Xs.append(X[idx_start:idx+1:])
        else:
            n_padding = self.window_size - idx - 1
            Xs = []
            for X in self.Xs:
                padding = np.repeat(X[0][None,:], n_padding, axis=0)
                X_origin = X[:idx+1]
                X_padded = np.vstack([X_origin, padding])
                Xs.append(X_padded)
                
        return Xs, self.y[idx]
    
    # def __getshape__(self):
    #     return (self.__len__(), *self.__getitem__(0)[0].shape)
    
    # def __getsize__(self):
    #     return (self.__len__())