import random
import ruleset
import json
import sys
import os
import argparse
import math


sys.setrecursionlimit(2000)
# random.seed(0) # Set the seed in real generation for reproducibility
def generate_random_sequence(grammar, symbol, max_depth, mode, depth = 0):
    if symbol in grammar:  # Check if the symbol is in the grammar
        production = None
        if depth >= max_depth:
            for p in grammar[symbol]:
                if len(p[0]) == 1:
                    production = p
                    p[1] = True
                    break
        if production is None:
            #print(symbol, grammar[symbol])
            production = random.choice(grammar[symbol])
            if (mode == "train"):
                production[1] = True
        return ''.join(generate_random_sequence(grammar, s, max_depth, mode, depth+1) for s in production[0])
    else:  # If the symbol is a terminal or a dummy symbol, return it
        #print("returning", symbol)
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

def generate_data(min_length, max_length, num_samples, grammar: ruleset.Ruleset, mode, train_data = None):
    result = {"pos": set(), "neg": set()}
    repeats = 0
    def inc_repeats():
        nonlocal repeats
        repeats += 1
        if repeats > 1000:
            raise "Maximum number of repeats (1000) was exceeded"
    for i in range(num_samples):
        l = 0
        pos_sample = ""
        neg_sample = ""
        while l < min_length or l > max_length or pos_sample in result["pos"] or (mode == "test" and pos_sample in train_data["pos"]):
            pos_sample = generate_random_sequence(grammar.rules, grammar.start_symbol, int(math.log2(max_length)) + 1, mode)
            l = len(pos_sample)
            inc_repeats()
        repeats = 0
        while neg_sample in result["neg"] or neg_sample == "" or (mode == "test" and neg_sample in train_data["neg"]):
            neg_sample = generate_invalid_sequence(grammar.symbols, grammar.rules, l, mode)
            inc_repeats()
        repeats = 0
        result["pos"].add(pos_sample)
        result["neg"].add(neg_sample)
    result["pos"] = list(result["pos"])
    result["neg"] = list(result["neg"])
    result["symbols"] = grammar.symbols
    return result

def create_dataset(min_train_length, max_train_length, min_test_length, max_test_length, num_train_samples, num_id_samples,
                   num_ood_samples, loc_grammar, loc_dataset_dir):
    grammar = ruleset.Ruleset()
    grammar.load(loc_grammar)

    train_data = generate_data(min_train_length, max_train_length, num_train_samples, grammar, "train")
    non_terminals = list(grammar.rules.keys())

    for nt in non_terminals:
        if (grammar.rules[nt] == []):
            grammar.rules.pop(nt)
            continue

    # omit unproductive rules from test sets
    for nt in non_terminals:
        length = len(grammar.rules[nt])
        removed = 0
        for i in range(length):
            rule = grammar.rules[nt][i-removed]
            if (not rule[1]):
                removed += 1
                grammar.rules[nt].remove(rule)

    # not sure how to export/save the data yet, will be completed later
    test_data = generate_data(min_train_length, max_train_length, num_id_samples, grammar, "test", train_data)
    ood_test = generate_data(min_test_length, max_test_length, num_ood_samples, grammar, "test", train_data)
    
    print(loc_dataset_dir)
    os.makedirs(loc_dataset_dir, exist_ok=True)
    with open(os.path.join(loc_dataset_dir, "train.json"), "w") as f:
        json.dump(train_data, f, indent=4)
    with open(os.path.join(loc_dataset_dir, "test_id.json"), "w") as f:
        json.dump(test_data, f, indent=4)
    with open(os.path.join(loc_dataset_dir, "test_ood.json"), "w") as f:
        json.dump(ood_test, f, indent=4)

if __name__ == "__main__":
    #create_dataset(1, 15, 200, 2500, 500, 250, "grammars/binary_tree/test.json", "datasets/binary_tree/")
    parser = argparse.ArgumentParser(description="Create a dataset with specified parameters")

    # Positional arguments
    parser.add_argument("min_train_length", type=int)
    parser.add_argument("max_train_length", type=int)
    parser.add_argument("min_test_length", type=int)
    parser.add_argument("max_test_length", type=int)
    parser.add_argument("num_train_samples", type=int)
    parser.add_argument("num_id_samples", type=int)
    parser.add_argument("num_ood_samples", type=int)
    parser.add_argument("loc_grammar", type=str)
    parser.add_argument("loc_dataset_dir", type=str)

    args = parser.parse_args()

    create_dataset(args.min_train_length, args.max_train_length, args.min_test_length, args.max_test_length,
                   args.num_train_samples, args.num_id_samples, args.num_ood_samples, args.loc_grammar,
                   args.loc_dataset_dir)