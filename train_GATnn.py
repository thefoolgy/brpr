import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
from sklearn.model_selection import random_split
from dataset.GAT_data import MolecularDataset
from GATnn import TransformationGAT
from transformers import BertTokenizer, BertModel
from pre_train_vae import VAE

dataset = MolecularDataset(root='./', filename='brenda_rm.csv')
train_dataset, test_dataset = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

vae = VAE()
vae.load_state_dict(torch.load('vae_model.pth'))
vae.eval()

# enzyme encoding
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
transformer_model = BertModel.from_pretrained('bert-base-uncased')
transformer_model.eval()

model = TransformationGAT(vae.decoder, input_dim=768, hidden_channels=128, transformation_dim=256, heads=4)
optimizer = optim.Adam(model.parameters(), lr=0.001)

criterion = torch.nn.MSELoss()

num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()

        # encode enzyme
        enzyme_tokens = tokenizer(data.enzyme_sequence, return_tensors="pt")
        enzyme_encoding = transformer_model(**enzyme_tokens).last_hidden_state.mean(dim=1)

        # encode molecule
        substrate_encoding = vae.encode(data.x)

        predicted_molecule = model(data, enzyme_encoding, substrate_encoding)

        # encode the true product
        true_product_encoding = vae.encode(data.y)

        loss = criterion(predicted_molecule, true_product_encoding)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

model.eval()
test_loss = 0
with torch.no_grad():
    for data in test_loader:
        enzyme_tokens = tokenizer(data.enzyme_sequence, return_tensors="pt")
        enzyme_encoding = transformer_model(**enzyme_tokens).last_hidden_state.mean(dim=1)

        substrate_encoding = vae.encode(data.x)

        predicted_molecule = model(data, enzyme_encoding, substrate_encoding)

        true_product_encoding = vae.encode(data.y)

        loss = criterion(predicted_molecule, true_product_encoding)
        test_loss += loss.item()

    print(f"Test Loss: {test_loss / len(test_loader)}")

torch.save(model.state_dict(), 'gat_model.pth')
