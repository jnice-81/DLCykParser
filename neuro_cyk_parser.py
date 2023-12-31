from logging import warn
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
import json
from torch import optim
import tqdm

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
        t = self.bt(t)# + t[0:t.shape[0] // 2] + t[t.shape[0] // 2:]
        t = self.b2(t)
        t = self.b22(t)
        return t

class NCykParser(nn.Module):
    def __init__(self, rule_count, symbols) -> None:
        super().__init__()

        self.prule = PRule(rule_count)
        self.tRules = nn.Embedding(len(symbols), rule_count)
        self.map = {sym: i for i, sym in enumerate(symbols)}
        self.topl = ResBlock(rule_count)
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
        result = self.ltopl(self.topl(emb))

        return result

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

ds = GrammarDataset("export2.json")
len_train_set = int(0.8 * len(ds))
test_ds, train_ds = data.random_split(ds, (len(ds) - len_train_set, len_train_set), torch.Generator().manual_seed(36))
dl_train = data.DataLoader(train_ds, 10, True)
dl_test = data.DataLoader(test_ds, 1, True)
model = NCykParser(4, ds.symbols)
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

for epoch in range(100):
    for _ in range(5):
        for sb, rb in tqdm.tqdm(dl_train):
            pred = torch.zeros(len(sb), 2, device=device)
            weights = torch.zeros(len(sb))
            for i, s in enumerate(sb):
                pred[i, :] = model(s)
                #weights[i] = 1 / len(s)
                weights[i] = 1
            rb = rb.long().to(device)
            loss = F.cross_entropy(pred, rb)
            #print(f"{pred} - {rb}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        
    with torch.no_grad():
        def compute_and_print_accuracy(dl, msg):
            count_correct = 0
            count_total = 0
            for sb, rb in dl:
                pred = torch.zeros(len(sb), 2)
                for i, s in enumerate(sb):
                    pred[i, :] = model(s)
                rb.to(device)
                count_total += len(sb)
                count_correct += (torch.argmax(pred, dim=1) == rb).sum()
            print(f"{msg} Acc: {count_correct / count_total}")
        compute_and_print_accuracy(dl_test, "test")
        compute_and_print_accuracy(dl_train, "train")
        
