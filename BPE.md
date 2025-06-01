Byte Pair Encoding (BPE) Algorithm — Study Notes
Purpose
BPE reduces vocabulary size by iteratively merging the most frequent adjacent symbol pairs in a corpus, creating meaningful subword tokens.

Complete Code
python
Copy
Edit
from collections import defaultdict, Counter  # Import data structures for counting frequencies

def get_stats(corpus):
    """Count all pairs of adjacent symbols in the corpus with their frequencies."""
    pairs = defaultdict(int)  # Dictionary to hold pair frequencies, default 0
    for word, freq in corpus.items():  # For each word and its frequency
        symbols = word.split()  # Split the word into symbols (characters/subwords)
        for i in range(len(symbols) - 1):  # For each adjacent pair in the word
            pairs[(symbols[i], symbols[i + 1])] += freq  # Increment count for this pair
    return pairs  # Return pair frequencies

def merge_vocab(pair, vocab):
    """Merge the most frequent pair in all words of the vocabulary."""
    new_vocab = {}  # New vocabulary after merging
    bigram = ' '.join(pair)  # Pair to merge as string, e.g. 'l o'
    replacement = ''.join(pair)  # Merged token, e.g. 'lo'
    for word in vocab:  # For each word in vocab
        new_word = word.replace(bigram, replacement)  # Replace all bigrams with merged token
        new_vocab[new_word] = vocab[word]  # Preserve frequency
    return new_vocab  # Return updated vocab

# Initial corpus with words split into characters + end token '</w>'
corpus = {
    'l o w </w>': 5,       # "low" frequency 5
    'l o w e r </w>': 2,   # "lower" frequency 2
    'l o w e s t </w>': 2, # "lowest" frequency 2
}

num_merges = 10  # Number of merge operations

for i in range(num_merges):
    pairs = get_stats(corpus)  # Count adjacent pairs
    if not pairs:  # Stop if no pairs left
        break
    best = max(pairs, key=pairs.get)  # Find most frequent pair
    print(f"Step {i+1}: merge {best}")  # Show merged pair
    corpus = merge_vocab(best, corpus)  # Merge pair in corpus
    print("New vocab:")
    for word in corpus:
        print(f"  {word}")  # Display updated vocabulary
Key Points to Remember
Corpus format: Words are split into symbols (characters or subwords) separated by spaces, ending with </w> to mark word boundaries.

get_stats: Counts frequency of every adjacent pair weighted by word frequency.

merge_vocab: Merges the selected pair throughout all words, updating vocabulary.

Loop: Repeats finding and merging the most frequent pair for num_merges iterations or until no pairs remain.

Output: After each merge, vocabulary tokens become larger subword units representing merged pairs.

This code is the core of BPE learning. Use it to understand how frequent pairs are merged stepwise to build an efficient subword vocabulary.
