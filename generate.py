# generate.py
# takes a trained network and a seed phrase
# predicts one word at a time, feeding each output back as input
# uses weighted random sampling so output isn't always identical
import numpy as np
from constants import TEMPERATURE
def generate(seed, num_words, network, word_to_idx, idx_to_word, n):
    words = seed.split()
    temperature = TEMPERATURE
    for _ in range(num_words):
        context = words[-n:]
        context_idx = [word_to_idx[w] for w in context]
        
        # we get raw scores before softmaxxing 
        # then apply temperature
        # lower : more conservative but robotic
        # higher: more exploratory but chaotic
        # mf keeps generating "kirk kirk kirk kirk" so
        # I'll experiment with this for a bit
        probs = network.forward(context_idx)
        logits = np.log(probs[0] + 1e-9) / temperature
        logits -= np.max (logits)
        exp_logits = np.exp (logits)
        probs_temp = exp_logits /exp_logits.sum()

        next_idx = np.random.choice(len(probs_temp), p= probs_temp)
        words.append(idx_to_word[next_idx])
    
    return " ".join(words)