import torch
from dataloader import CFGDataset
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import tqdm

class TransformerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super(TransformerClassifier, self).__init__()

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=1),
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # print(x.shape)
        x = self.embedding(x)
        # print(x.shape)
        x = self.transformer_encoder(x)
        # print(x.shape)
        x = x.mean(dim=1)  # Aggregate across the sequence dimension
        # print(x.shape)
        x = self.fc(x)
        # x = torch.softmax(x, dim=0)
        return x

# Hyperparameters
input_size = 3  # 'a', 'b', and padding
hidden_size = 32
num_classes = 2  # Binary classification (0 or 1)
num_layers = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = "datasets/random/test_ood.json"

batchsize = 1

training_data = CFGDataset(path, "train")
train_dataloader = DataLoader(dataset=training_data, batch_size=batchsize, shuffle=True)

test_id_data = CFGDataset(path, "test_id")
test_id_dataloader = DataLoader(dataset=test_id_data, batch_size=batchsize, shuffle=True)

test_ood_data = CFGDataset(path, "test_ood")
test_ood_dataloader = DataLoader(dataset=test_ood_data, batch_size=batchsize, shuffle=True)

model = TransformerClassifier(input_size, hidden_size, num_classes, num_layers).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1
for epoch in range(epochs):
    for inputs, targets in tqdm.tqdm(train_dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

# Test the model
correct = 0
with torch.no_grad():
    for inputs, targets in test_id_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        test_outputs = model(inputs)
        predictions = torch.argmax(test_outputs)
        correct += (predictions == targets).item()

print(f"ID accuracy: {correct / len(test_id_data)}")

# Test the model
correct = 0
with torch.no_grad():
    for inputs, targets in test_ood_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        test_outputs = model(inputs)
        predictions = torch.argmax(test_outputs)
        correct += (predictions == targets).item()

print(len(test_ood_data))
print(len(test_id_data))
print(f"OOD accuracy: {correct / len(test_ood_data)}")
