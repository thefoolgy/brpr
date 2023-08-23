import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class VAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super(VAE, self).__init__()
        self.encoder_conv = GCNConv(in_channels, hidden_channels)
        self.encoder_mu = torch.nn.Linear(hidden_channels, latent_dim)
        self.encoder_logvar = torch.nn.Linear(hidden_channels, latent_dim)
        self.decoder = torch.nn.Linear(latent_dim, in_channels)

    def encode(self, x, edge_index):
        x = F.relu(self.encoder_conv(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.encoder_mu(x), self.encoder_logvar(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar * 0.5)
            q = torch.distributions.Normal(mu, std)
            return q.rsample()
        else:
            return mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
