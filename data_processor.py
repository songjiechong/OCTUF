import torch
import platform
import h5py
import scipy.io as sio

def data_loader(args):
    if (platform.system() == "Windows"):
        num_workers = 0
    else:
        num_workers = 4
    kwopt = {'num_workers': num_workers, 'pin_memory': True}

    Training_data_Name = 'traindata0-255.mat'
    f = h5py.File('./DataSets/%s' % Training_data_Name, 'r')
    Training_data = f['inputs'][:]
    Training_lable = Training_data

    class RandomDataset(torch.utils.data.Dataset):
        def __init__(self, data, length):
            self.data = data
            self.len = length

        def __getitem__(self, index):
#             print("self.data:", self.data.shape)
            return torch.Tensor(self.data[index, :, :, :]).float()

        def __len__(self):
            return self.len

    trn_loader = torch.utils.data.DataLoader(dataset=RandomDataset(Training_lable, 89600), batch_size=args.batch_size,
                                             shuffle=True, **kwopt, drop_last=False)

    return trn_loader
