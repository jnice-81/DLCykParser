import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
import json
from torch import optim
import tqdm
import csv
import os
import sys

class ProdRule(nn.Module):
    def __init__(self, rule_count) -> None:
        super().__init__()

        self.W = nn.Parameter(torch.normal(1, 0.1, (rule_count, rule_count)))
    
    def forward(self, u, v):
        x = torch.outer(u, v)
        x = self.W * x
        x = torch.max(x)

        return x
    
class NCykParser(nn.Module):
    def __init__(self, rule_count, symbols) -> None:
        super().__init__()

        self.pRules = nn.ModuleList([ProdRule(rule_count) for _ in range(rule_count)])
        self.tRules = nn.Embedding(len(symbols), rule_count)
        nn.init.normal_(self.tRules.weight, 1, 0.1)
        self.map = {sym: i for i, sym in enumerate(symbols)}

    def apply_rule(self, s):
        if s in self.cache:
            return self.cache[s]
        else:
            r = self.intern_forward(s)
            self.cache[s] = r
            return r
        
    def intern_forward(self, s: str):
        if len(s) == 1:
            return self.tRules(torch.tensor(self.map[s], device=self.tRules.weight.device))
        else:
            result = torch.zeros(len(self.pRules), device=self.tRules.weight.device)
            for i in range(1, len(s)):

                u = self.apply_rule(s[:i])
                v = self.apply_rule(s[i:])

                for j, p in enumerate(self.pRules):
                    r = p(u, v)
                    if r > result[j]:
                        result[j] = r
            return result

    def forward(self, s: str):
        self.cache = {}

        result = self.intern_forward(s)

        return result[0].unsqueeze(0)

class GrammarDataset(data.Dataset):
    def __init__(self, file, type) -> None:
        super().__init__()

        with open(file, "r") as f:
            l = json.load(f)
        l = l[type]
        self.pos = l["pos"]
        self.neg = l["neg"]
        self.symbols = l["symbols"]

    def __len__(self):
        return len(self.pos) + len(self.neg)

    def __getitem__(self, index): # Use with shuffle
        if index >= len(self.pos):
            return self.neg[index - len(self.pos)], torch.tensor(0.0, dtype=torch.float32)
        else:
            return self.pos[index], torch.tensor(1.0, dtype=torch.float32)
        
def compute_and_log_accuracy(dl, msg):
    count_correct = 0
    count_total = 0
    for sb, rb in tqdm.tqdm(dl, msg):
        pred = torch.zeros(len(sb), 2)
        for i, s in enumerate(sb):
            pred[i, :] = model(s)
        rb.to(device)
        count_total += len(sb)
        count_correct += (torch.argmax(pred, dim=1) == rb).sum()
    acc = count_correct / count_total
    print(f"{msg} Acc: {acc}")
    return acc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

base_folder = sys.argv[1]
num_rules = int(sys.argv[2])
logfilename = f"ncykv1({num_rules} rules).csv"

train_ds = GrammarDataset(os.path.join(base_folder, "data.json"), "train")
test_ds = GrammarDataset(os.path.join(base_folder, "data.json"), "test_id")
ood_ds = GrammarDataset(os.path.join(base_folder, "data.json"), "test_ood")
dl_train = data.DataLoader(train_ds, 1, True)
dl_test = data.DataLoader(test_ds, 1, True)
dl_test_ood = data.DataLoader(ood_ds, 1, True)
model = NCykParser(num_rules, train_ds.symbols)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

with open(os.path.join(base_folder, logfilename), "w", newline='') as log:
    csv_writer = csv.writer(log)
    csv_writer.writerow(["valid", "train", "ood"])

for epoch in range(10):
    for _ in range(2):
        for sb, rb in tqdm.tqdm(dl_train):
            pred = torch.zeros(len(sb))
            weights = torch.zeros(len(sb))
            for i, s in enumerate(sb):
                pred[i] = model(s)
                #weights[i] = 1 / len(s)
                weights[i] = 1
            rb.to(device)
            loss = torch.sum((torch.abs((1 - rb) * pred) + torch.abs(rb * (pred - 2))) * weights)
            #loss = torch.binary_cross_entropy_with_logits(pred - 1, rb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            break
        
    with torch.no_grad():
        for p in model.pRules:
            p.W[p.W < 0] = 0
        model.tRules.weight[model.tRules.weight < 0] = 0

        acc_test = compute_and_log_accuracy(dl_test, "valid")
        acc_train = compute_and_log_accuracy(dl_train, "train")
        acc_ood = compute_and_log_accuracy(dl_test_ood, "ood")
        with open(os.path.join(base_folder, logfilename), "a", newline='') as log:
            csv_writer = csv.writer(log)
            csv_writer.writerow([acc_test.item(), acc_train.item(), acc_ood.item()])
        