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
        
        # remove words not in vocab
        context = [w for w in context if w in word_to_idx]
        if len(context) == 0:
            break

        while len(context) < n:
            context = [context[-1]] + context
        
        context_idx = [word_to_idx[w] for w in context]
        
        # we get raw scores before softmaxxing 
        # then apply temperature
        # lower : more conservative but robotic
        # higher: more exploratory but chaotic
        # mf keeps generating "kirk kirk kirk kirk" so
        # I'll experiment with this for a bit
        probs = network.forward(context_idx)[0]
        
        # smoother temp scaling for now since its a small model
        probs = probs ** (1 / temperature)
        probs = probs / probs.sum()

        # repetition penalty
        # prevents bot ass sentences despite fixed overfitting prevention
        recent_words = words[-5:]  # last few generated words

        for i in range(len(probs)):
            word = idx_to_word[i]
            count= recent_words.count(word)
            if count >0:
                probs[i]*=(0.5 ** count) # lower-> more aggressive exponential penalty
               

        probs = probs / probs.sum()

        # top k filtering
        # at this point, we have prob distribution over all words in the vocab
        # this is why output is consistently acoustic like "dumb they him it"
        # I'll try to restrict sampling to only top kek most likely used words

        k=8
        top_indices = np.argsort(probs)[-k:]
        mask= np.zeros_like(probs)
        mask[top_indices] = 1

        probs = probs * mask
        probs /= probs.sum()

        # tiny prob smoothing
        # adds randomness prevents one token from dominating
        probs += 1e-9
        probs /= probs.sum()

        next_idx = np.random.choice(len(probs), p=probs)
        words.append(idx_to_word[next_idx])
    
    return " ".join(words)