import torch
from dataloader import CFGDataset
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

class TransformerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers, nhead):
        super(TransformerClassifier, self).__init__()

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead),
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, mask=None):
        # print(x.shape)
        x = self.embedding(x)
        # print(x.shape)
        x = self.transformer_encoder(x)
        # print(x.shape)
        x = x.mean(dim=1)  # Aggregate across the sequence dimension
        # print(x.shape)
        x = self.fc(x)
        # print(x)
        x = torch.softmax(x, dim=1)
        # print(x)
        return x

# Hyperparameters
input_size = 4  # 'a', 'b', and padding
hidden_size = 16
nhead = 1
num_classes = 2  # Binary classification (0 or 1)
num_layers = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = "test_dataset.json"

dataset = CFGDataset(path)
train_dataset, test_dataset = random_split(dataset, [1600, 400])
trainloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
model = TransformerClassifier(input_size, hidden_size, num_classes, num_layers, nhead).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
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
