import random
import ruleset

# random.seed(0) # Set the seed in real generation for reproducibility
def generate_random_sequence(grammar, symbol):
    if symbol in grammar:  # Check if the symbol is in the grammar
        production = random.choice(grammar[symbol])
        return ''.join(generate_random_sequence(grammar, s) for s in production)
    else:  # If the symbol is a terminal or a dummy symbol, return it
        return symbol

def generate_invalid_sequence(valid_symbols, length):
    invalid_sequence = ''.join(random.choice(valid_symbols) for _ in range(length))
    return invalid_sequence

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
                        if len(production) == 2:
                            B, C = production
                            if B in table[start][mid] and C in table[mid + 1][end]:
                                table[start][end].add(A)
    
    # If the start symbol is in the top-right corner of the table, the sequence is in the language
    start_symbol = list(grammar.keys())[0]
    return start_symbol in table[0][n - 1]

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
