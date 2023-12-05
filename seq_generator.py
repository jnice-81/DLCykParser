import random

# random.seed(0) # Set the seed in real generation for reproducibility

def generate_random_sequence(grammar, symbol):
    if symbol in grammar:  # Check if the symbol is in the grammar
        production = random.choice(grammar[symbol])
        print(f"Production: {production}")
        return ''.join(generate_random_sequence(grammar, s) for s in production)
    else:  # If the symbol is a terminal, return it
        return symbol

def main():
    cnf_grammar = {
        'S': ["AE","BF"],
        'G': ["GG","a","b","AB"],
        'E': ["GA","a"],
        'F': ["GB","b"],
        'A': ["a"],
        'B': ["b"]
    }

    # Set the start symbol
    start_symbol = 'S'

    # Generate a random sequence
    random_sequence = generate_random_sequence(cnf_grammar, start_symbol)

    print(f"Random Sequence: {random_sequence}")

if __name__ == "__main__":
    main()