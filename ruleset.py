import json

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

