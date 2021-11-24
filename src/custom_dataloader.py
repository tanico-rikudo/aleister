import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, *tensors, **params):
        self.X_tensors = tensors[:-1]
        self.y_tensor = tensors[-1]
        self.sequence_length = params["sequence_length"]
        self.batch_size = params["batch_size"]
        self.Xs = X_tensors.float()
        self.y = y_tensor.float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx): 
        if idx >= self.sequence_length - 1:
            idx_start = idx - self.sequence_length + 1
            Xs = []
            for X in self.X:
                Xs.append(X[idx_start:idx+1:])
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return Xs, self.y[i]