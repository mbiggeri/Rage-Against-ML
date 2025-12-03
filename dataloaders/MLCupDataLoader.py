import torch
from torch.utils.data import Dataset, DataLoader

class MLCupDataLoader(DataLoader):
  def __init__(self, dataset, batch_size, shuffle=True):
    super(MLCupDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle)

class MLCupDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class MLCupBlindDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]