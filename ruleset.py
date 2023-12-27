import json
import random

class Ruleset:
    def __init__(self) -> None:
        self.rules = {}
        self.start_symbol = None
        self.symbols = []

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)

        self.start_symbol = data["start_symbol"]
        self.symbols = data["symbols"]

        for rule in data["rules"]:
            from_symbol = rule["From"]
            to_symbols = rule["To"]
            # add a field for tracking whether the rule was productive during generation
            to_symbols = track_rules(to_symbols)
            self.rules[from_symbol] = to_symbols

    def as_json(self):
        data = {
            "start_symbol": self.start_symbol,
            "rules": [],
            "symbols": self.symbols
        }

        for from_symbol, to_symbols in self.rules.items():
            rule = {"From": from_symbol, "To": to_symbols}
            data["rules"].append(rule)

        return data

    def write(self, path: str):
        data = self.as_json()

        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def __str__(self) -> str:
        return json.dumps(self.as_json(), indent=4)

def track_rules(to_symbols):
    return [[rule, False] for rule in to_symbols]

def generate_random_ruleset(valid_symbols, max_rules, max_different_rules, seed=0):
    random.seed(seed)

    rules = set()

    for i, s in enumerate(valid_symbols):
        rules.add((i, (s, )))

    for _ in range(max_rules - len(valid_symbols)):
        rules.add((random.randint(0, max_different_rules), 
                   (random.randint(0, max_different_rules),
                    random.randint(0, max_different_rules))))
    

    # Based on this: https://www.cs.scranton.edu/~mccloske/courses/cmps260/cfg_remove_useless.html

    productive_rules = set()
    productive_names = set()
    old_size = -1

    while old_size < len(productive_rules):
        old_size = len(productive_rules)
        for rule in rules:
            if (
                isinstance(rule[1][0], str) or
                (rule[1][0] in productive_names and rule[1][1] in productive_names)
                ):
                productive_names.add(rule[0])
                productive_rules.add(rule)

    rs = Ruleset()
    for rulename, to in productive_rules:
        rs.rules.setdefault(rulename, [])
        rs.rules[rulename].append(to)

    rs.start_symbol = random.choice(list(productive_names))
    
    reachable = set()
    query = set([rs.start_symbol])

    while len(query) > 0:
        u = query.pop()
        reachable.add(u)
        for r in rs.rules[u]:
            if not isinstance(r[0], str):
                for nq in r:
                    if nq not in reachable and nq not in query:
                        query.add(nq)

    diff = set()
    for r in rs.rules:
        if r not in reachable:
            diff.add(r)
    for r in diff:
        rs.rules.pop(r)

    for _, l in rs.rules.items():
        for g in l:
            if isinstance(g[0], str):
                rs.symbols.append(g[0])

    return rs
