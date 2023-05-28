import torch
from torch.utils.data import Dataset
import scipy.io as scio


class ecgdata(Dataset):
    def __init__(self, data_root):
        super(ecgdata, self).__init__()
        self.data_root = data_root
        self.datas = []
        self.labels = []
        for index in range(4000):
            data_file = data_root + f'/{index}.mat'
            data = scio.loadmat(data_file)
            self.datas.append(data['ecg'])
            self.labels.append(data['label'])

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        data = self.datas[item]
        label = self.labels[item]
        return data,label

if __name__ == '__main__':
    