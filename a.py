import json
import os

def read(f):
    with open(f, "r") as fd:
        return json.load(fd)
    
def write(o, f):
    with open(f, "w") as fd:
        json.dump(o, fd, indent=4)
    
def compose(base):
    train = read(os.path.join(base, "train.json"))
    test_id = read(os.path.join(base, "test_id.json"))
    test_ood = read(os.path.join(base, "test_ood.json"))

    g = {}
    g["train"] = train
    g["test_id"] = test_id
    g["test_ood"] = test_ood

    write(g, os.path.join(base, "data.json"))

compose("datasets/random/grammar3/small_ds")