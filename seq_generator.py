import random
import ruleset
import json
import sys


sys.setrecursionlimit(20000)
# random.seed(0) # Set the seed in real generation for reproducibility
def generate_random_sequence(grammar, symbol, max_depth, mode, depth = 0):
    if symbol in grammar:  # Check if the symbol is in the grammar
        production = None
        if depth >= max_depth:
            for p in grammar[symbol]:
                if not isinstance(p[0], list):
                    production = p
                    break
        if production is None:
            production = random.choice(grammar[symbol])
            if (mode == "train"):
                production[1] = True
        return ''.join(generate_random_sequence(grammar, s, max_depth, mode, depth+1) for s in production[0])
    else:  # If the symbol is a terminal or a dummy symbol, return it
        return symbol

def cyk_parse(sequence, grammar):
    n = len(sequence)
    non_terminals = list(grammar.keys())
    
    # Initialize the table
    table = [[set() for _ in range(n)] for _ in range(n)]
    
    # Fill in the diagonal of the table based on the terminals in the grammar
    for i in range(n):
        for nt in non_terminals:
            if sequence[i] in grammar[nt]:
                table[i][i].add(nt)
    
    # Fill in the rest of the table using the CYK algorithm
    for length in range(2, n + 1):
        for start in range(n - length + 1):
            end = start + length - 1
            for mid in range(start, end):
                for A in non_terminals:
                    for production in grammar[A]:
                        production = production[0]
                        if len(production) == 2:
                            B, C = production
                            if B in table[start][mid] and C in table[mid + 1][end]:
                                table[start][end].add(A)
    
    # If the start symbol is in the top-right corner of the table, the sequence is in the language
    start_symbol = list(grammar.keys())[0]
    return start_symbol in table[0][n - 1]

def generate_invalid_sequence(valid_symbols, grammar_rules, length, mode):
    is_valid = True
    while is_valid:
        invalid_sequence = ''.join(random.choice(valid_symbols)[0] for _ in range(length))
        is_valid = cyk_parse(invalid_sequence, grammar_rules)
    return invalid_sequence

def generate_data(min_length, max_length, num_samples, grammar: ruleset.Ruleset, mode):
    result = {"pos": set(), "neg": set()}
    repeats = 0
    def inc_repeats():
        nonlocal repeats
        repeats += 1
        print(repeats)
        if repeats > 1000:
            raise "Maximum number of repeats (1000) was exceeded"
    for i in range(num_samples):
        l = 0
        pos_sample = ""
        neg_sample = ""
        while l < min_length or l > max_length or pos_sample in result["pos"]:
            pos_sample = generate_random_sequence(grammar.rules, grammar.start_symbol, max_length // 2, mode)
            l = len(pos_sample)
            inc_repeats()
        repeats = 0
        while neg_sample in result["neg"] or neg_sample == "":
            neg_sample = generate_invalid_sequence(grammar.symbols, grammar.rules, l, mode)
            inc_repeats()
        repeats = 0
        result['pos'].add(pos_sample)
        result["neg"].add(neg_sample)
    result["pos"] = list(result["pos"])
    result["neg"] = list(result["neg"])
    result["symbols"] = grammar.symbols
    return result

def create_dataset(min_length, max_train_length, max_test_length, num_train_samples, num_id_samples,
                   num_ood_samples, loc_grammar, loc_dataset):
    grammar = ruleset.Ruleset()
    grammar.load(loc_grammar)

    train_data = generate_data(min_length, max_train_length, num_train_samples, grammar, "train")
    with open(loc_dataset, "w") as f:
        json.dump(train_data, f, indent=4)

    non_terminals = list(grammar.rules.keys())

    # omit unproductive rules from test sets
    for nt in non_terminals:
        for rule in grammar.rules[nt]:
            if (not rule[1]):
                grammar.rules[nt].remove(rule)
    
    print("lol")
    # not sure how to export/save the data yet, will be completed later
    ood_test = generate_data(max_train_length+1, max_test_length, num_ood_samples, grammar, "test")
    test_data = generate_data(min_length, max_train_length, num_id_samples, grammar, "test")

#create_dataset(1, 10, 200, "test_rules.json", "export.json")

"""
def main():
    # Define your grammar in CNF
    cnf_grammar = ruleset.Ruleset()
    cnf_grammar.load("test_rules.json")
    # cnf_grammar = {
    #     'S': ["AE","BF"],
    #     'G': ["GG","a","b","AB"],
    #     'E': ["GA","a"],
    #     'F': ["GB","b"],
    #     'A': ["a"],
    #     'B': ["b"]
    # }


    # Generate a random valid sequence
    random_sequence = generate_random_sequence(cnf_grammar.rules, cnf_grammar.start_symbol)
    print(f"Random Valid Sequence: {random_sequence}")

   # Generate a random invalid sequence
    invalid_sequence = generate_invalid_sequence(cnf_grammar.symbols, len(random_sequence))
    while cyk_parse(invalid_sequence, cnf_grammar.rules):    
        invalid_sequence = generate_invalid_sequence(cnf_grammar.symbols, len(random_sequence))
    
    print(f"Random Invalid Sequence: {invalid_sequence}")

if __name__ == "__main__":
    main()
"""

if __name__ == "__main__":
    create_dataset(2, 75, 100, 1600, 300, 100, "test_rules.json", "kekw.json")
# create_dataset(2, 100, 1000, "test_rules.json", "test_dataset.json")