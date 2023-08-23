import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class TransformationGAT(torch.nn.Module):
    def __init__(self, vae_decoder, input_dim, hidden_channels, transformation_dim, heads):
        super(TransformationGAT, self).__init__()

        # VAE Decoder
        self.vae_decoder = vae_decoder

        # GAT layers
        self.gat_conv1 = GATConv(input_dim, hidden_channels, heads=heads, dropout=0.6)
        self.gat_conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=0.6)

        # Final transformation layer
        self.transformation_fc = torch.nn.Linear(hidden_channels * heads, transformation_dim)

    def forward(self, data, enzyme_encoding, substrate_encoding):
        # Concatenate enzyme and substrate encodings
        combined_encoding = torch.cat([enzyme_encoding, substrate_encoding], dim=1)

        # Use the combined encoding as node features
        x = combined_encoding
        edge_index = data.edge_index

        x = F.elu(self.gat_conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.elu(self.gat_conv2(x, edge_index))

        # Global pooling (mean over all nodes)
        global_representation = torch.mean(x, dim=0, keepdim=True)

        # Transformation to predict the product's representation
        predicted_product_representation = self.transformation_fc(global_representation)

        # use vae_decoder
        predicted_molecule = self.vae_decoder(predicted_product_representation)

        return predicted_molecule