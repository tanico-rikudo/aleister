import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset

import numpy  as np


class SequenceDataset(Dataset):
    def __init__(self, *tensors, **params):
        self.X_tensors = tensors[:-1]
        self.y_tensor = tensors[-1]
        self.window_size = params["window_size"]
        self.batch_size = params["batch_size"]  # keep
        self.Xs = [X_tensor.numpy() for X_tensor in self.X_tensors]
        self.y = self.y_tensor.numpy()

    def __len__(self):
        return self.Xs[0].shape[0]

    def __getitem__(self, idx):
        if idx >= (self.window_size - 1):
            idx_start = idx - self.window_size + 1
            Xs = []
            for X in self.Xs:
                Xs.append(X[idx_start:idx + 1:])
        else:
            n_padding = self.window_size - idx - 1
            Xs = []
            for X in self.Xs:
                padding = np.repeat(X[0][None, :], n_padding, axis=0)
                X_origin = X[:idx + 1]
                X_padded = np.vstack([X_origin, padding])
                Xs.append(X_padded)

        return Xs, [self.y[idx]]


class CustomTensorDataset(TensorDataset):

    def __init__(self, *data, **params):
        super(CustomTensorDataset, self).__init__(*data)
        # data = data[0]
        if isinstance(data, dict):
            assert len(data) > 0, "Should have at least one element"
            # check that all fields have the same size
            n_elem = len(list(data.values())[0])
            for v in data.values():
                assert len(v) == n_elem, "All values must have the same size"
        elif isinstance(data, list):
            assert len(data) > 0, "Should have at least one element"
            n_elem = len(data[0])
            for v in data:
                assert len(v) == n_elem, "All elements must have the same size"

        self.data = data

    def __len__(self):
        return self.data[0].shape[0]
        # if isinstance(self.data, dict):
        #     return len(list(self.data.values())[0])
        # elif isinstance(self.data, list):
        #     return len(self.data[0])
        # elif torch.is_tensor(self.data) or isinstance(self.data, np.ndarray):
        #     return len(self.data)

    def __getitem__(self, idx):
        return [self.data[0][idx]], [self.data[1][idx]]

        # if isinstance(self.data, dict):
        #     return {k: [v[idx]] for k, v in self.data.items()}
        # elif isinstance(self.data, list):
        #     return [[v[idx]] for v in self.data]
        # elif torch.is_tensor(self.data) or isinstance(self.data, np.ndarray):
        #     return [self.data[idx]]
