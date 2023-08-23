import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Dataset, Data
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader


class MolecularDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MolecularDataset, self).__init__(root, transform, pre_transform)
        if not self.processed_exists():
            self.data = pd.read_csv(self.raw_paths[0])
            self.process()

    @property
    def raw_file_names(self):
        return ['brenda_rm.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        self.data['substrate_mol'] = self.data['smiles'].apply(self.smiles_to_mol)
        self.data['product_mol'] = self.data['product_smile'].apply(self.smiles_to_mol)
        self.data['substrate_atom_features'] = self.data['substrate_mol'].apply(
            lambda mol: [self.atom_feature(atom) for atom in mol.GetAtoms()])
        self.data['substrate_bond_features'] = self.data['substrate_mol'].apply(
            lambda mol: [self.bond_feature(bond) for bond in mol.GetBonds()])
        self.data['product_atom_features'] = self.data['product_mol'].apply(
            lambda mol: [self.atom_feature(atom) for atom in mol.GetAtoms()])
        self.data['product_bond_features'] = self.data['product_mol'].apply(
            lambda mol: [self.atom_feature(bond) for bond in mol.GetBonds()])
        self.data['substrate_graph'] = self.data.apply(
            lambda row: self.molecule_to_graph(
                row['substrate_mol'], row['substrate_atom_features'], row['substrate_bond_features'])
            , axis=1)
        self.data['product_graph'] = self.data.apply(
            lambda row: self.molecule_to_graph(
                row['product_mol'], row['product_atom_features'], row['product_bond_features'])
            , axis=1)
        self.data['graph_difference'] = self.data.apply(
            lambda row: self.compute_graph_difference(row['substrate_graph'], row['product_graph']), axis=1)

    @staticmethod
    def smiles_to_mol(smiles_str):
        return Chem.MolFromSmiles(smiles_str)

    @staticmethod
    def atom_feature(atom):
        return [
            atom.GetAtomicNum(),
            atom.GetTotalValence(),
            atom.GetDegree(),
            atom.IsInRing(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons()
        ]

    @staticmethod
    def bond_feature(bond):
        bond_type = bond.GetBondType()
        return [
            bond_type == Chem.rdchem.BondType.SINGLE,
            bond_type == Chem.rdchem.BondType.DOUBLE,
            bond_type == Chem.rdchem.BondType.TRIPLE,
            bond_type == Chem.rdchem.BondType.AROMATIC,
            bond.IsInRing()
        ]

    @staticmethod
    def molecule_to_graph(mol, atom_features, bond_features):
        x = torch.tensor(atom_features, dtype=torch.float)

        bond_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_indices.append((start, end))
            bond_indices.append((end, start))
            edge_attrs.extend([bond_features[bond.GetIdx()]] * 2)

        edge_index = torch.tensor(bond_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    @staticmethod
    def compute_graph_difference(substrate_graph, product_graph):
        # Node difference
        substrate_nodes = set(tuple(node) for node in substrate_graph.x.tolist())
        product_nodes = set(tuple(node) for node in product_graph.x.tolist())

        added_nodes = product_nodes - substrate_nodes
        removed_nodes = substrate_nodes - product_nodes

        # Edge difference
        substrate_edges = set(tuple(edge) for edge in substrate_graph.edge_index.t().tolist())
        product_edges = set(tuple(edge) for edge in product_graph.edge_index.t().tolist())

        added_edges = product_edges - substrate_edges
        removed_edges = substrate_edges - product_edges

        return {
            "added_nodes": added_nodes,
            "removed_nodes": removed_nodes,
            "added_edges": added_edges,
            "removed_edges": removed_edges
        }

    def len(self):
        return len(self.data)

    def get(self, idx):
        substrate_graph = self.data.iloc[idx]['substrate_graph']
        product_graph = self.data.iloc[idx]['product_graph']
        graph_difference = self.data.iloc[idx]['graph_difference']
        return substrate_graph, product_graph, graph_difference


def get_dataloaders(batch_size=32, val_size=0.1):
    dataset = MolecularDataset(root='./')
    train_data, temp_data = train_test_split(dataset, test_size=0.3)
    val_data, test_data = train_test_split(temp_data,test_size=2 / 3)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

