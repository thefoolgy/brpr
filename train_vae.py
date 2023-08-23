import torch
from torch_geometric.data import DataLoader
from torch.optim import Adam
from dataset.VAE_data import MoleculeDataset
from pre_train_vae import VAE

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

model = VAE(in_channels=34, hidden_channels=256, latent_dim=64)
optimizer = Adam(model.parameters(), lr=0.001)

dataset = MoleculeDataset(root='./', csv_file='brenda_rm.csv')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

def train(epoch):
    model.train()
    train_loss = 0
    for data in dataloader:
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data.x, data.edge_index)
        loss = loss_function(recon_batch, data.x, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(dataloader.dataset)))

num_epochs = 500
for epoch in range(1, num_epochs + 1):
    train(epoch)
