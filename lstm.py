import os
import torch
import numpy as np 

from dataloader import CFGDataset
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.autograd import Variable 




SEED = 2024
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

batchsize = 25 # how many samples the network sees before it updates itself
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = "datasets/random/test_ood.json"

training_data = CFGDataset(path, "train")
train_dataloader = DataLoader(dataset=training_data, batch_size=batchsize, shuffle=True)

test_id_data = CFGDataset(path, "test_id")
test_id_dataloader = DataLoader(dataset=test_id_data, batch_size=batchsize, shuffle=True)

test_ood_data = CFGDataset(path, "test_ood")
test_ood_dataloader = DataLoader(dataset=test_ood_data, batch_size=batchsize, shuffle=True)



input_dim = 200 #max length of sequence
hidden_size = 512
num_layers = 2
num_classes = 2
num_epochs = 500
learning_rate = 0.002


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes, num_layers, batchsize):
        super(LSTM, self).__init__()
        #seq_length = embedding_dim
        #self.embedding = nn.Embedding(batchsize, input_dim) #batchsize x sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
        self.batchsize = batchsize
        self.output_layer = nn.Linear(hidden_size, num_classes) #fully connected last layer
        self.dropout = nn.Dropout(0.1)
        

    def forward(self,x):
        #x = self.embedding(x)
        x = self.dropout(x)
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state
        # Propagate input through LSTM
        #print("SHAPE OF X befor self.lstm: " ,x.shape)
        out, _ = self.lstm(x, (h_0, c_0))  # (batchsize x sequencelength x nb_features), (tuple)
        
        out = self.output_layer(out[:, -1, :]) #flatten output
       
        return out
 
    def init_hidden(self, x):
        h0 = torch.zeros(self.num_layers,  x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers,  x.size(0), self.hidden_size)
        return [t.cuda() for t in (h0, c0)]

model = LSTM(input_dim, hidden_size, num_classes, num_layers, batchsize).to(device)
#print(model)


loss_func = nn.CrossEntropyLoss()
#optimzer
sgd = optim.SGD(model.parameters(), lr=learning_rate)
adam = optim.Adam(model.parameters(), lr=learning_rate)


def train(num_epochs, model, train_dataloader, loss_func, arr):
    total_steps = len(train_dataloader)
    correct = 0
    for epoch in range(num_epochs):
        
        
        for batch, (x_batch, y_batch) in enumerate(train_dataloader):
            #x_batch must have dim 3: torch.Size([50, 1, 3]) (additional param timestamp)
            #y_batch must have dim 2: torch.Size([50, 1])
            #print("Training Shape", x_batch.shape, y_batch.shape)
            x_batch = torch.reshape(x_batch,(x_batch.shape[0], 1, x_batch.shape[1])).to(device).float()
            #y_batch = y_batch.reshape(batchsize, 1).to(device)
            y_batch = y_batch.to(device)
            #print("Training Shape after reshape", x_batch.shape, y_batch.shape)
            adam.zero_grad()
            output = model(x_batch)
            loss = loss_func(output, y_batch)

            
            loss.backward()
            adam.step()
            #to have a better/more comparable starting number for the loss and accuracy plot(after one batch it would otherwise already have accuracy 80% which is tru,e but compared to the nureocyk it 
            #technically did way more after one epoch, so i just put it like that)
            if(batch == 1 and (epoch==0 or epoch == 1)):
                _ , a_train = test_loop(train_dataloader, model, loss_func, adam)
                _ , a_id = test_loop(test_id_dataloader, model, loss_func, adam)
                _, a_ood = test_loop(test_ood_dataloader, model, loss_func, adam)
                arr[0][epoch] = a_train
                arr[1][epoch] = a_id
                arr[2][epoch] = a_ood


        if(epoch > 1):
            _, a_train = test_loop(train_dataloader, model, loss_func, adam)
            #for plots, to have accuracy after each epoch:
            _, a_id = test_loop(test_id_dataloader, model, loss_func, adam)
            _, a_ood = test_loop(test_ood_dataloader, model, loss_func, adam)
            arr[0][epoch] = a_train
            arr[1][epoch] = a_id
            arr[2][epoch] = a_ood

    return arr

def test_loop(dataloader, model, loss_func, optimizer):
    size= len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for x, y in dataloader:
            #reshape X = X.respahe..
            x = torch.reshape(x,(x.shape[0], 1, x.shape[1])).to(device).float()
            #y_batch = y_batch.reshape(batchsize, 1).to(device)
            y = y.to(device)
            pred = model(x)
            test_loss += loss_func(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error:\n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")
    return test_loss, 100*correct

arr = np.empty((3, num_epochs)) #test on the same dataset as trianing -> 100% to overfit
arr = train(num_epochs, model, train_dataloader, loss_func, arr)
#on training set
test_loop(train_dataloader, model, loss_func, adam)
#validatio set
test_loop(test_id_dataloader, model, loss_func, adam)

test_loop(test_ood_dataloader, model, loss_func, adam)
#np.savetxt('lstm_csvForPlot/random_test_ood.csv', arr, delimiter=',', header='train,valid,ood', comments='')