import random
from ruleset import Ruleset
import argparse

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a random ruleset with specified parameters")

    parser.add_argument("--valid_symbols", nargs='+', required=True)
    parser.add_argument("--max_rules", type=int, required=True)
    parser.add_argument("--max_different_rules", type=int, required=True)
    parser.add_argument("--loc", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # Call the new function
    ruleset = generate_random_ruleset(args.valid_symbols, args.max_rules, args.max_different_rules, args.seed)
    ruleset.write(args.loc)

    print(f"num rules {len(ruleset.rules)}")
    print(f"num productions {sum([1 if len(g) == 2 else 0 for prods in ruleset.rules.values() for g in prods])}")
    print(f"num terminal productions {sum([1 if len(g) == 1 else 0 for prods in ruleset.rules.values() for g in prods])}")