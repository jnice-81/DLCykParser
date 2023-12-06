import json

class Ruleset:
    def __init__(self) -> None:
        self.rules = {}
        self.start_symbol = None

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)

        self.start_symbol = data["start_symbol"]
        self.symbols = data["symbols"]

        for rule in data["rules"]:
            from_symbol = rule["From"]
            to_symbols = rule["To"]
            self.rules[from_symbol] = to_symbols

    def write(self, path: str):
        data = {
            "start_symbol": self.start_symbol,
            "rules": [],
            "symbols": self.symbols
        }

        for from_symbol, to_symbols in self.rules.items():
            rule = {"From": from_symbol, "To": to_symbols}
            data["rules"].append(rule)

        with open(path, "w") as f:
            json.dump(data, f, indent=4)