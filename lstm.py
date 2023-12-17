import os
import torch
import numpy as np 

from dataloader import CFGDataset
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

batchsize = 50 # how many samples the network sees before it updates itself
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = "test_dataset.json"
dataset = CFGDataset(path)
training_data, test_data = random_split(dataset, [1600, 400])

train_dataloader = DataLoader(dataset=training_data, batch_size=batchsize, shuffle=True)
test_dataloader =  DataLoader(dataset=test_data, batch_size=batchsize, shuffle=True)

#define hyperparameters
input_dim = 3 
hidden_size = 128
num_layers = 2
num_classes = 2
num_epochs = 10
learning_rate = 0.01

print(training_data)

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes, num_layers, batchsize):
        super(LSTM, self).__init__()
        ##for 3d x 3d in self.lstm
        self.embedding = nn.Embedding(input_dim, hidden_size)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batchsize, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, num_classes)
        self.batchsize = batchsize


        
    def forward(self, X):
        #init hidden states
        print(X.shape)
        X = self.embedding(X)
        h0, c0 = self.init_hidden(X)
        out, _ = self.lstm(X, (h0, c0))
        
        out = self.output_layer(out[:, -1, :]) #flatten output
       
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.num_layers,  x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers,  x.size(0), self.hidden_size)
        return [t.cuda() for t in (h0, c0)]

model = LSTM(input_dim, hidden_size, num_classes, num_layers, batchsize)
print(model)


loss_func = nn.CrossEntropyLoss()
#optimzer
sgd = optim.SGD(model.parameters(), lr=learning_rate)
adam = optim.Adam(model.parameters(), lr=learning_rate)


def train(num_epochs, model, train_dataloader, loss_func):
    total_steps = len(train_dataloader)
    for epoch in range(num_epochs):
        for batch, (x_batch, labels) in enumerate(train_dataloader):
            #respahe into right size
            
            #x_batch = x_batch.reshape(batchsize, len(train_dataloader) + 1, input_dim).to(device)
            x_batch = x_batch.to(device)
            labels = labels.to(device)
            sgd.zero_grad()
            output = model(x_batch)
            loss = loss_func(output, labels)

            
            loss.backward()
            sgd.step()

            if(batchsize+1)%100 == 0:
                print(f"Epoch: {epoch+1}; Batch{batchsize+1} / {total_steps}; Loss: {loss.item():>4f}")


def test_loop(dataloader, model, loss_func, optimizer):
    size= len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            #reshape X = X.respahe..
            pred = model(X)
            test_loss += loss_func(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error:\ Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")
    return 100*correct


train(num_epochs, model, train_dataloader, loss_func)

test_loop(test_dataloader, model, loss_func, sgd)

