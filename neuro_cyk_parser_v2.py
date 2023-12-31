import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
import json
from torch import optim
import tqdm
import os
import csv
import sys

SEED = 2024
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class ResBlock(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        l = []
        scale_dim = 3 * dim
        l.append(nn.Linear(dim, scale_dim))
        l.append(nn.LeakyReLU())
        l.append(nn.Linear(scale_dim, dim))
        l.append(nn.LeakyReLU())
        l.append(nn.Linear(dim, scale_dim))
        l.append(nn.LeakyReLU())
        l.append(nn.Linear(scale_dim, dim))
        l.append(nn.LeakyReLU())
        self.block = nn.Sequential(*l)

    def forward(self, x):
        return self.block(x) + x
    
class PRule(nn.Module):
    def __init__(self, rule_count) -> None:
        super().__init__()

        self.b1 = ResBlock(2 * rule_count)
        self.b11 = ResBlock(2 * rule_count)
        self.bt = nn.Linear(2 * rule_count, rule_count)
        self.b2 = ResBlock(rule_count)
        self.b22 = ResBlock(rule_count)

    def norm(self, t):
        return (t - t.mean()) / torch.var(t)

    def forward(self, x, y):
        t = torch.cat((x, y))
        t = self.b1(t)
        t = self.b11(t)
        t = self.bt(t)
        t = self.b2(t)
        t = self.b22(t)
        return t

class NCykParser(nn.Module):
    def __init__(self, rule_count, symbols) -> None:
        super().__init__()

        self.prule = PRule(rule_count)
        self.tRules = nn.Embedding(len(symbols), rule_count)
        self.map = {sym: i for i, sym in enumerate(symbols)}
        self.ltopl = nn.Linear(rule_count, 2)

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
            results = []
            for i in range(1, len(s)):

                u = self.apply_rule(s[:i])
                v = self.apply_rule(s[i:])

                comb = self.prule(u, v)
                results.append(comb)
            r = torch.stack(results, 1)
            r, _ = torch.max(r, dim=1)
            return r

    def forward(self, s: str):
        self.cache = {}

        emb = self.intern_forward(s)
        result = self.ltopl(emb)

        return result

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

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

base_folder = sys.argv[1]
num_rules = int(sys.argv[2])
if len(sys.argv) > 3:
    epoch_count = int(sys.argv[3])
else:
    epoch_count = 10
logfilename = f"ncykv2({num_rules} rules).csv"

train_ds = GrammarDataset(os.path.join(base_folder, "data.json"), "train")
test_ds = GrammarDataset(os.path.join(base_folder, "data.json"), "test_id")
ood_ds = GrammarDataset(os.path.join(base_folder, "data.json"), "test_ood")
dl_train = data.DataLoader(train_ds, 1, True)
dl_test = data.DataLoader(test_ds, 1, True)
dl_test_ood = data.DataLoader(ood_ds, 1, True)
model = NCykParser(num_rules, train_ds.symbols)
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

with open(os.path.join(base_folder, logfilename), "w", newline='') as log:
    csv_writer = csv.writer(log)
    csv_writer.writerow(["valid", "train", "ood"])

for epoch in range(epoch_count):
    model.train()
    for sb, rb in tqdm.tqdm(dl_train):
        pred = torch.zeros(len(sb), 2, device=device)
        for i, s in enumerate(sb):
            pred[i, :] = model(s)
        rb = rb.long().to(device)
        loss = F.cross_entropy(pred, rb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
            
    model.eval()
    with torch.no_grad():
        torch.save(model.state_dict(), os.path.join(base_folder, f"chpt_{epoch}.pth"))
        acc_test = compute_and_log_accuracy(dl_test, "valid")
        acc_train = compute_and_log_accuracy(dl_train, "train")
        acc_ood = compute_and_log_accuracy(dl_test_ood, "ood")
        with open(os.path.join(base_folder, logfilename), "a", newline='') as log:
            csv_writer = csv.writer(log)
            csv_writer.writerow([acc_test.item(), acc_train.item(), acc_ood.item()])

log.close()