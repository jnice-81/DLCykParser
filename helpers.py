import json

class Ruleset:
    def __init__(self) -> None:
        self.prod_rules = {}
        self.term_rules = {}
        self.start_symbol = None

    def load(self, path: str):
        with open(path) as f:
            pjson = json.load(f)

        for prod_rule in pjson["prod_rules"]:
            self.prod_rules[prod_rule["From"]] = prod_rule["To"]

        for term_rule in pjson["term_rules"]:
            self.term_rules[term_rule["From"]] = term_rule["To"]

        self.start_symbol = pjson["start_symbol"]

    def write(self, path: str):
        pjson = {}

        jsonprod = []
        for from_rule, to in self.prod_rules.items():
            g = {}
            g["From"] = from_rule
            g["To"] = to
            jsonprod.append(g)

        jsonterm = []
        for from_rule, to in self.term_rules.items():
            g = {}
            g["From"] = from_rule
            g["To"] = to
            jsonterm.append(g)

        pjson["prod_rules"] = jsonprod
        pjson["term_rules"] = jsonterm
        pjson["start_symbol"] = self.start_symbol

        with open(path, "w") as f:
            json.dump(pjson, f, indent=4)

r = Ruleset()
r.load("test_rules.json")
r.write("out.json")