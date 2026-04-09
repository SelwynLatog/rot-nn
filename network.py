# network.py
# takes word indices -> converts to embedding vectors + positional encoding (concatenated n-gram)
# forward pass: x -> (xW1 + b1) -> ReLU -> (hW2 + b2) -> softmax -> probabilities
# loss: compares prediction with correct word (cross-entropy)
# backward: computes gradients and updates weights
import numpy as np
from constants import N,HIDDEN_SIZE, LEARNING_RATE, WEIGHT_SCALE, EMBED_DIM_SIZE
class net:
    def __init__(self, vocab_size, hidden_size, n):
        # vocab_size : INPUT
        #              how many words we have
        # hidden_size: HIDDEN LAYER
        #              neurons in the middle layer
        # vocab_size : OUTPUT neurons scores for each prediction word
        # added n    : larger input size

        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.n = n

        # weights
        # w1: input -> hidden   shape: (EMBED_DIM_SIZE * n, hidden_size)
        # w2: hidden -> output  shape: (hidden_size, vocab_size)
        self.w1 = np.random.randn(EMBED_DIM_SIZE * n, hidden_size) * WEIGHT_SCALE
        self.w2 = np.random.randn(hidden_size, vocab_size) * WEIGHT_SCALE
        self.e  = np.random.randn(vocab_size, EMBED_DIM_SIZE) * WEIGHT_SCALE
        
        # added positional encoding. sin cos fixed slot
        self.p  = np.zeros((n,EMBED_DIM_SIZE))
        for pos in range (n):
            for i in range (EMBED_DIM_SIZE):
                if i % 2 == 0 :
                    self.p[pos, i] = np.sin(pos / (10000 ** (i / EMBED_DIM_SIZE)))
                else:
                    self.p[pos, i] = np.cos(pos / (10000 ** (i / EMBED_DIM_SIZE)))  

        # biases start at 0, one per neuron per layer
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, vocab_size))

    def embed(self, word_idx):
        # builds input vector from n-gram:
        # each word → embedding + positional encoding
        # returns concatenated vector of shape (1, EMBED_DIM_SIZE * N)
        vec = []
        for i , idx in enumerate(word_idx):
            # previous:
            # vecN= np.zeros((1, self.vocab_size))
            # vecN[0, idx] = 1

            # current:
            # embedding + positional encoding
            embed = self.e[idx] + self.p[i]     # add position signal
            embed = embed.reshape(1, -1)        # shape (1, EMBED_DIM_SIZE)
            vec.append(embed)
        return np.concatenate (vec, axis=1) # shape (1, EMBED_DIM_SIZE *N)

    def relu(self, x):
        # rectified linear unit
        # negative gets bonked to 0, positve unchanged
        return np.maximum(0, x)

    def softmax(self, x):
        # output layer activation
        # raw scores -> probabilities that sum to 1
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def forward(self, word_indices):
        # prev implementation using one_hot vec:
        # word_indices -> one_hot -> ×w1 + b1 -> relu -> ×w2 + b2 -> softmaxxing -> probs
        # x = self.one_hot(word_indices)  # shape: (1, vocab_size * N)
        
        # current using embedding lookup + positional encoding:
        self.input_indices = word_indices # save for backprop

        x= self.embed(word_indices) # returns concatenated embeddings
        self.x= x

        self.z1 = np.dot(x, self.w1)+self.b1
        self.a1 = self.relu (self.z1)
        self.z2 = np.dot(self.a1, self.w2)+ self.b2
        self.probs = self.softmax(self.z2)

        return self.probs

    def loss(self, probs, target_idx):
        # cross entropy loss
        # low confidence on correct answer  -> high loss
        # high confidence on correct answer -> low loss
        correct_prob = probs[0, target_idx]
        return -np.log(correct_prob + 1e-9)  # 1e-9 prevents log(0) crash

    def backward(self, target_idx, learning_rate=LEARNING_RATE):
        # backpropagation
        # trace error backward, compute gradients, nudge all weights

        # output layer gradients
        # subtract 1 from correct word's slot — bigger error = bigger signal
        dz2 = self.probs.copy()
        dz2[0, target_idx] -= 1                 # shape: (1, vocab_size)

        dw2 = np.dot(self.a1.T, dz2)            # shape: (hidden_size, vocab_size)
        db2 = dz2                               # shape: (1, vocab_size)

        # hidden layer gradients
        # pass error back through w2
        da1 = np.dot(dz2, self.w2.T)            # shape: (1, hidden_size)

        # pass back through relu
        dz1 = da1 * (self.z1 > 0)               # shape: (1, hidden_size)

        dw1 = np.dot(self.x.T, dz1)             # shape: (EMBED_DIM_SIZE * N, hidden_size)
        db1 = dz1                               # shape: (1, hidden_size)

        # nudge every weight
        # new_weight = old_weight - learning_rate × gradient
        self.w2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2
        self.w1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1

        # embedding update
        # NOTE: positional encoding (self.p) is fixed -> not updated during backprop
        dx = np.dot(dz1, self.w1.T)
        for i, idx in enumerate(self.input_indices):
            self.e[idx] -= learning_rate * dx[0, i * EMBED_DIM_SIZE : (i+1) * EMBED_DIM_SIZE]

# test one cycle
if __name__ == "__main__":
    from data import build_vocab, get_ngram_pairs, sentences

    word_to_idx, idx_to_word = build_vocab(sentences)
    vocab_size = len(word_to_idx)

    network  = net(vocab_size=vocab_size, hidden_size=HIDDEN_SIZE, n=N) # change to whatever n size and hidden_size
    pairs    = get_ngram_pairs(sentences, word_to_idx, n=N)
    inp, tgt = pairs[0]

    input_words = [idx_to_word[i] for i in inp]
    print(f"input words  : {input_words}")
    print(f"correct next : '{idx_to_word[tgt]}'")

    # before learning
    probs = network.forward(inp)
    l     = network.loss(probs, tgt)
    print(f"\nbefore backward pass:")
    print(f"  probability on correct word : {probs[0, tgt]:.4f}")
    print(f"  loss                        : {l:.4f}")

    # one backward pass
    network.backward(tgt, learning_rate=LEARNING_RATE)

    # after one nudge
    probs = network.forward(inp)
    l     = network.loss(probs, tgt)
    print(f"\nafter one backward pass:")
    print(f"  probability on correct word : {probs[0, tgt]:.4f}")
    print(f"  loss                        : {l:.4f}")
    