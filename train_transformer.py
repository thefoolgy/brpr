from pre_train_transformer import ProteinBertTokenizer, BertForMaskedLM, tokens
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import torch.optim as optim


# Load your protein sequences
def load_protein_sequences(filename):
    sequences = []
    for record in SeqIO.parse(filename, "fasta"):
        sequences.append(str(record.seq))
    return sequences


protein_sequences = load_protein_sequences("dataset/uniprot_sprot.fasta")
tokenizer = ProteinBertTokenizer(tokens=tokens, vocab_file=None)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Tokenize sequences
inputs = tokenizer(protein_sequences, return_tensors="pt", padding=True, truncation=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
dataloader = DataLoader(inputs, batch_size=32, shuffle=True, collate_fn=data_collator)

optimizer = optim.AdamW(model.parameters(), lr=5e-5)
model.train()

for epoch in range(10):  # Adjust the number of epochs
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
