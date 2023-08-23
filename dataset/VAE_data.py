import torch
from torch_geometric.data import Dataset, Data
import pandas as pd


class MoleculeDataset(Dataset):
    def __init__(self, root, csv_file, transform=None, pre_transform=None):
        self.csv_file = csv_file
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [self.csv_file]

    def len(self):
        self.data = pd.read_csv(self.raw_paths[0])
        return len(self.data)

    def get(self, idx):
        row = self.data.iloc[idx]
        x = torch.tensor(row['x'], dtype=torch.float)
        edge_index = torch.tensor(row['edge_index'], dtype=torch.long)
        return Data(x=x, edge_index=edge_index)
