# generate.py
# takes a trained network and a seed phrase
# predicts one word at a time, feeding each output back as input
# uses weighted random sampling so output isn't always identical
import numpy as np

def generate(seed, num_words, network, word_to_idx, idx_to_word, n):
    words = seed.split()
    
    for _ in range(num_words):
        context = words[-n:]
        context_idx = [word_to_idx[w] for w in context]
        
        probs = network.forward(context_idx)
        next_idx = np.random.choice(len(probs[0]), p=probs[0])
        next_word = idx_to_word[next_idx]
        
        words.append(next_word)
    
    return " ".join(words)