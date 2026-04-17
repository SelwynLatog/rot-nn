# generate.py
# takes a trained network and a seed phrase
# predicts one word at a time, feeding each output back as input
# uses weighted random sampling so output isn't always identical
import numpy as np
from constants import TEMPERATURE, N
def generate(seed, num_words, network, word_to_idx, idx_to_word, n):
    words = seed.split()
    for _ in range(num_words):
        
        context = words[-n:]
        
        # remove words not in vocab
        context = [w for w in context if w in word_to_idx]
        if len(context) == 0:
            break

        while len(context) < n:
            context = [context[0]] + context
        
        context_idx = [word_to_idx[w] for w in context]
        
        # we get raw scores before softmaxxing 
        # then apply temperature
        # lower : more conservative but robotic
        # higher: more exploratory but chaotic
        # mf keeps generating "kirk kirk kirk kirk" so
        # I'll experiment with this for a bit
        probs = network.forward(context_idx)[0].copy()
        
        # smoother temp scaling for now since its a small model
        # temp in log space
        log_probs  = np.log(probs + 1e-9) / TEMPERATURE
        log_probs -= np.max(log_probs)
        probs = np.exp(log_probs)
        probs/= probs.sum()

        # TOP K FILTERING
        k=15
        top_indices= np.argsort(probs)[-k:]
        mask= np.zeros_like(probs)
        mask[top_indices] =1
        probs= probs * mask
        probs /= probs.sum()

        # hard exclusion window
        recent_words = set(words[-4:])
        for i in range (len(probs)):
            if idx_to_word[i] in recent_words:
                probs[i]=0.0
        
        # fallback block
        if probs.sum() == 0:
            mask        = np.zeros_like(probs)
            mask[top_indices] = 1
            probs  = network.forward(context_idx)[0].copy() * mask
            probs /= probs.sum()
        else:
            probs /= probs.sum()

        next_idx = np.random.choice(len(probs),p=probs)
        words.append(idx_to_word[next_idx])
       
    
    return " ".join(words)