import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class TransformationGAT(torch.nn.Module):
    def __init__(self, vae_decoder, input_dim, hidden_channels, transformation_dim, heads):
        super(TransformationGAT, self).__init__()

        self.vae_decoder = vae_decoder

        self.gat_conv1 = GATConv(input_dim, hidden_channels, heads=heads, dropout=0.6)
        self.gat_conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=0.6)

        self.transformation_fc = torch.nn.Linear(hidden_channels * heads, transformation_dim)

    def forward(self, data, enzyme_encoding, substrate_encoding):
        enzyme_encoding = F.elu(self.gat_conv1(enzyme_encoding, edge_index))
        enzyme_encoding = F.dropout(enzyme_encoding, training=self.training)
        enzyme_encoding = F.elu(self.gat_conv2(enzyme_encoding, edge_index))

        substrate_encoding = F.elu(self.gat_conv1(substrate_encoding, edge_index))
        substrate_encoding = F.dropout(substrate_encoding, training=self.training)
        substrate_encoding = F.elu(self.gat_conv2(substrate_encoding, edge_index))

        combined_encoding = torch.cat([enzyme_encoding, substrate_encoding], dim=1)
        x = combined_encoding
        edge_index = data.edge_index

        global_representation = torch.mean(x, dim=0, keepdim=True)

        predicted_product_representation = self.transformation_fc(global_representation)

        # use vae_decoder
        predicted_molecule = self.vae_decoder(predicted_product_representation)

        return predicted_molecule
