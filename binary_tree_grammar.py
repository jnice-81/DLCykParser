from numpy.random import uniform 

class Tree():
    def __init__(self, left = None, right = None, label = None, depth = None):
        self.depth = depth
        self.left = left
        self.right = right
        self.label = label

# used to assign unique labels, as all recursive calls refer to the same object
class Pointer:
    def __init__(self, val):
        self.val = val


def increment_string(s):
    if not s:
        return 'a'
    
    reversed_s = s[::-1]
    carry = 1
    result = []

    for char in reversed_s:
        if carry:
            if char == 'z':
                result.append('a')
            else:
                result.append(chr(ord(char) + 1))
                carry = 0
        else:
            result.append(char)

    if carry:
        result.append('a')

    return ''.join(result[::-1])

# generates a binary tree with exactly the specified depth and given branching probability
# on average, the right hand side of the tree with respect to the root tends to be deeper
# but this should not matter as the production rules are permutation invariant
def generate_tree(depth, prob, index, root = True):
    curr_index = int(index.val)
    index.val += 1

    if (depth == 0):
        return Tree(label = curr_index, depth = 0)

    if (not root):
        sample = uniform()

        if (sample > 1-prob):
            left = generate_tree(depth - 1, prob, index, False)
            right = generate_tree(depth - 1, prob, index, False)
            return Tree(left, right, curr_index, max(left.depth, right.depth) + 1)
        else:
            return Tree(label = curr_index, depth = 0)
    
    left = generate_tree(depth - 1, prob, index, False)

    if (left.depth == depth - 1):
        right = generate_tree(depth - 1, prob, index, False)
    else:
        right = generate_tree(depth - 1, prob, index, True)

    return Tree(left, right, curr_index, max(left.depth, right.depth) + 1)
    

def parse_tree(tree, rules, terminator, symbols):
    curr_symbol = str(terminator.val)
    terminator.val = increment_string(terminator.val)
    
    if (tree.left is None):
        entry = {"From": tree.label, "To": [curr_symbol]}
        symbols.append(str(symbols))
        rules.append(entry)
        return
    
    entry = {"From": tree.label, "To": [[tree.left.label, tree.right.label], curr_symbol]}
    symbols.append(str(symbols))
    rules.append(entry)

    parse_tree(tree.left, rules, terminator, symbols)
    parse_tree(tree.right, rules, terminator, symbols)

def create_grammar(depth = 3, prob = 1):
    tree = generate_tree(depth, prob, Pointer(0), True)
    rules = []
    symbols = []
    parse_tree(tree, rules, Pointer('a'), symbols)

    return {"start_symbol": tree.label, "rules": rules, "symbols": symbols}

if __name__ == "__main__":
    tree = generate_tree(3, 1, Pointer(0), True)
    rules = []
    symbols = []
    parse = parse_tree(tree, rules, Pointer('a'), symbols)
    
