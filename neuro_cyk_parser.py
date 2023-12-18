from logging import warn
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
import json
from torch import optim

class ProdRule(nn.Module):
    def __init__(self, rule_count) -> None:
        super().__init__()

        self.W = nn.Parameter(torch.normal(1, 0.1, (rule_count, rule_count)))
    
    def forward(self, u, v):
        x = torch.outer(u, v)
        x = self.W * x
        x = F.relu(x)
        x = torch.max(x)

        return x
    
class NCykParser(nn.Module):
    def __init__(self, rule_count, symbols) -> None:
        super().__init__()

        self.pRules = nn.ModuleList([ProdRule(rule_count) for _ in range(rule_count)])
        self.tRules = nn.Embedding(len(symbols), rule_count)
        nn.init.normal_(self.tRules.weight, 1, 0.1)
        self.map = {sym: i for i, sym in enumerate(symbols)}

        """
        with torch.no_grad():
            warn("Debug initialization")
            for i in range(len(self.pRules)):
                self.pRules[i].W = nn.Parameter(torch.zeros((rule_count, rule_count)))
            self.pRules[0].W[1, 2] = 1.0
            self.pRules[0].W[2, 3] = 1.0
            self.pRules[1].W[2, 1] = 1.0
            self.pRules[2].W[3, 3] = 1.0
            self.pRules[3].W[1, 2] = 1.0
            nn.init.zeros_(self.tRules.weight)
            self.tRules.weight[0, 1] = 1.0
            self.tRules.weight[0, 3] = 1.0
            self.tRules.weight[1, 2] = 1.0
            pass
        """

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
            result = torch.zeros((len(self.pRules), len(s) - 1), device=self.tRules.weight.device)
            for i in range(1, len(s)):

                u = self.apply_rule(s[:i])
                v = self.apply_rule(s[i:])

                for j, p in enumerate(self.pRules):
                    r = p(u, v)
                    result[j, i-1] = r
            a, _ = torch.max(result, dim=1)
            return a

    def forward(self, s: str):
        self.cache = {}

        result = self.intern_forward(s)

        return result[0].unsqueeze(0)

class GrammarDataset(data.Dataset):
    def __init__(self, file) -> None:
        super().__init__()

        with open(file, "r") as f:
            l = json.load(f)
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

ds = GrammarDataset("export.json")
test_ds, train_ds = data.random_split(ds, (0.2, 0.8))
dl_train = data.DataLoader(train_ds, 1, True)
dl_test = data.DataLoader(test_ds, 1, True)
model = NCykParser(4, ds.symbols)
model2 = NCykParser(4, ds.symbols)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(60):
    for sb, rb in dl_train:
        pred = torch.zeros(len(sb))
        for i, s in enumerate(sb):
            pred[i] = model(s)
        rb.to(device)
        loss = torch.sum(torch.abs((1.0 - rb) * pred) + torch.abs(rb * (pred - 2)))
        
        loss.backward()
        
        """
        for p in model.pRules:
            print(p.W.grad)
        print(model.tRules.weight.grad)
        print(f"{rb.item()} - {pred.item()}")
        """
        
        #print(f"{r.item()} - {pred.item()}")
        optimizer.step()
        optimizer.zero_grad()
        
    with torch.no_grad():
        for p in model.pRules:
            p.W[p.W < 0] = 0
        model.tRules.weight[model.tRules.weight < 0] = 0

        """
        for p, w in zip(model.pRules, model2.pRules):
            w.W = p.W.clone()
            w.W[w.W < 1] = 0
            w.W[w.W > 1] = 1
            print(p.W)
        model2.tRules.weight = model.tRules.weight.clone()
        model2.tRules.weight[model2.tRules.weight < 1] = 0
        model2.tRules.weight[model2.tRules.weight > 1] = 1
        print(model.tRules.weight)
        """

        count_correct = 0
        count_total = 0
        for sb, rb in dl_test:
            pred = torch.zeros(len(sb))
            for i, s in enumerate(sb):
                pred[i] = model2(s)
            rb.to(device)
            count_total += len(sb)
            count_correct += torch.logical_or(torch.logical_and(pred < 1, rb == 0), torch.logical_and( pred >= 1, rb == 1)).sum()
        print(f"Acc: {count_correct / count_total}")

for p in model.pRules:
    print(p.W)
print(model.tRules.weight)
print(model.map)