import torch

class PyODDataset(torch.utils.data.Dataset):
    """PyOD Dataset class for PyTorch Dataloader
    """

    def __init__(self, X, y=None, mean=None, std=None):
        super(PyODDataset, self).__init__()
        self.X = X
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()#将tensor类型转换列表格式
        sample = self.X[idx, :]

        # if self.mean.any():
        #     sample = (sample - self.mean) / (self.std + 1e-5)
        #torch.from_numpy()将numpy类型转换为tensor类型
        return torch.from_numpy(sample), idx
