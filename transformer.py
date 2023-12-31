import torch
from dataloader import CFGDataset
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

SEED = 2024
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
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

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
        )
        self.linear = nn.Linear(d_model, num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Aggregate across the sequence dimension
        x = self.linear(x)
        return x

# Hyperparameters
input_size = 4  # 'a', 'b', and padding
d_model = 16
dim_feedforward = 64
nhead = 1
num_classes = 2  # Binary classification (0 or 1)
num_layers = 1
dropout = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = "test_dataset.json"

dataset = CFGDataset(path)
train_dataset, test_dataset = random_split(dataset, [1600, 400])
trainloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
model = TransformerClassifier(input_size, d_model, num_classes, num_layers, nhead, dim_feedforward, dropout).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
epochs = 100
epoch_losses = []
for epoch in range(epochs):
    batch_loss = []
    for inputs, targets in tqdm(trainloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())

    epoch_loss = sum(batch_loss) / len(batch_loss)
    epoch_losses.append(sum(batch_loss) / len(batch_loss))
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

# Plot the loss curve
plt.plot(epoch_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Test the model
correct = 0
with torch.no_grad():
    for inputs, targets in trainloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        test_outputs = model(inputs)
        predictions = torch.argmax(test_outputs, dim=1)
        correct += (predictions == targets).sum().item()

print(f"Accuracy in training set: {correct / len(train_dataset)}")

correct = 0
with torch.no_grad():
    for inputs, targets in test_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        test_outputs = model(inputs)
        predictions = torch.argmax(test_outputs)
        correct += (predictions == targets).item()

print(f"Accuracy in test set: {correct / len(test_dataset)}")
