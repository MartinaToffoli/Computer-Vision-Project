import torch
from torch.utils import data
import pandas as pd

class MyDataSet(data.Dataset):
    def __init__(self, filename):
        x = pd.read_csv(filename).iloc[:, 1:].values
        y = pd.read_csv(filename).iloc[:, 0].values

        self.X_train = torch.tensor(x, dtype=torch.float32).reshape(-1, 1, 28, 28)
        self.Y_train = torch.tensor(y)
    def __len__(self):
        return len(self.Y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.Y_train[idx]
