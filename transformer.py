import torch
from dataloader import CFGDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import os

SEED = 2024
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # print(self.pe.shape, x.shape)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(self, input_size, d_model, num_classes, num_layers, nhead, dim_feedforward, dropout):
        super(TransformerClassifier, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(input_size, d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
        
        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Aggregate across the sequence dimension
        x = self.linear(x)
        return x

# Hyperparameters
input_size = 27 # 26 letters + <pad>
d_model = 50
dim_feedforward = 50
nhead = 5
num_classes = 2  # Binary classification (0 or 1)
num_layers = 6
dropout = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# path = "test_dataset.json"
path = os.path.join("datasets", "random", "grammar1", "large_ds", "data.json")

model = TransformerClassifier(input_size, d_model, num_classes, num_layers, nhead, dim_feedforward, dropout).to(device)

training_data = CFGDataset(path, "train")
train_dataloader = DataLoader(dataset=training_data, batch_size=40, shuffle=True)

test_id_data = CFGDataset(path, "test_id")
test_id_dataloader = DataLoader(dataset=test_id_data, batch_size=10, shuffle=True)

test_ood_data = CFGDataset(path, "test_ood")
test_ood_dataloader = DataLoader(dataset=test_ood_data, batch_size=10, shuffle=True)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
model.train()
epochs = 8000
epoch_losses = []
for epoch in range(epochs):
    batch_loss = []
    for inputs, targets in tqdm(train_dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())

    epoch_loss = sum(batch_loss) / len(batch_loss)
    epoch_losses.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

# Plot the loss curve
plt.plot(epoch_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Test the model
model.eval()
correct = 0
with torch.no_grad():
    for inputs, targets in train_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        test_outputs = model(inputs)
        predictions = torch.argmax(test_outputs, dim=1)
        correct += (predictions == targets).sum().item()

print(f"Accuracy in training set: {correct / len(training_data)}")

correct = 0
with torch.no_grad():
    for inputs, targets in test_id_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        test_outputs = model(inputs)
        predictions = torch.argmax(test_outputs, dim=1)
        correct += (predictions == targets).sum().item()

print(f"ID accuracy: {correct / len(test_id_data)}")

# Test the model
correct = 0
with torch.no_grad():
    for inputs, targets in test_ood_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        test_outputs = model(inputs)
        predictions = torch.argmax(test_outputs, dim=1)
        correct += (predictions == targets).sum().item()

print(f"OOD accuracy: {correct / len(test_ood_data)}")