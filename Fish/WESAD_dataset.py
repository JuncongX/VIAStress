import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

data = np.load('data/WESAD/WESAD_clip30s_multi_peak808_.npy', allow_pickle=True)


# data = np.load('../data/WESAD/WESAD_clip30s_multi_peak808_.npy', allow_pickle=True)


def z_score(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    return (signal - mean) / (std + 1e-8)


class WESAD_DomainDataset(Dataset):
    def __init__(self, person, binary=True):
        needed_data = None
        for label in [0, 1, 2]:
            selected_data = data[(data[:, 0] == person) & (data[:, 1] == label)]
            if needed_data is None:
                needed_data = selected_data
            else:
                needed_data = np.concatenate((needed_data, selected_data))

        if binary:
            self.labels = torch.from_numpy(np.array([(l if l == 1 else 0) for l in needed_data[:, 1]]).astype(np.uint8))
        else:
            self.labels = torch.from_numpy(np.array([l for l in needed_data[:, 1]]).astype(np.uint8))

        self.PPG = torch.Tensor([z_score(sig).astype(np.float32) for sig in needed_data[:, 2]])
        self.EDA = torch.Tensor([z_score(sig).astype(np.float32) for sig in needed_data[:, 5]])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        x = (self.PPG[idx].unsqueeze(dim=0).to(torch.float), self.EDA[idx].unsqueeze(dim=0).to(torch.float))
        y = self.labels[idx].long()
        return x, y


if __name__ == '__main__':
    from Fishr.fast_data_loader import InfiniteDataLoader

    persons = [r"S{0}".format(i) for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]]

    domain_datasets = [WESAD_DomainDataset(person) for person in persons]

    domain_loaders = []
    for dataset in domain_datasets:
        loader = InfiniteDataLoader(
            dataset=dataset,
            weights=None,
            batch_size=128,
            num_workers=1,
        )
        domain_loaders.append(iter(loader))

    total = 0
    for i in range(3):
        minibatches = []
        for loader in domain_loaders:
            batch = next(loader)
            x, y = batch
            minibatches.append((x, y))

        print(len(minibatches))
        for x, y in minibatches:
            ppg, eda = x
            print(ppg.shape)
            total += ppg.shape[0]
            print(eda.shape)
            print(y)
            # break
    print(total)
