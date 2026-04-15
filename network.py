# network.py
# takes word indices -> converts to embedding vectors + positional encoding (concatenated n-gram)
# forward pass: x -> (xW1 + b1) -> ReLU -> (hW2 + b2) -> softmax -> probabilities
# loss: compares prediction with correct word (cross-entropy)
# backward: computes gradients and updates weights
import numpy as np
from constants import N,HIDDEN_SIZE, LEARNING_RATE, WEIGHT_SCALE, EMBED_DIM_SIZE, ATTN_DIM_SIZE, NUM_HEADS
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
        self.w1 = np.random.randn(ATTN_DIM_SIZE * n, hidden_size) * WEIGHT_SCALE
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

        # prev:
        # attention layer implementation
        # Query , Key , Value
        # self.Wq = np.random.randn(EMBED_DIM_SIZE, ATTN_DIM_SIZE) * WEIGHT_SCALE
        # self.Wk = np.random.randn(EMBED_DIM_SIZE, ATTN_DIM_SIZE) * WEIGHT_SCALE
        # self.Wv = np.random.randn(EMBED_DIM_SIZE, ATTN_DIM_SIZE) * WEIGHT_SCALE

        # curr:
        # Multi head expansion
        # New Query, Key, Value
        self.num_heads = NUM_HEADS
        self.head_dim_size= ATTN_DIM_SIZE // NUM_HEADS # eg.gives 16//4=4 
        self.Wq = np.random.randn(self.num_heads, EMBED_DIM_SIZE, self.head_dim_size) * WEIGHT_SCALE
        self.Wk = np.random.randn(self.num_heads, EMBED_DIM_SIZE, self.head_dim_size) * WEIGHT_SCALE
        self.Wv = np.random.randn(self.num_heads, EMBED_DIM_SIZE, self.head_dim_size) * WEIGHT_SCALE

        # shape output projection
        self.Wo = np.random.randn(ATTN_DIM_SIZE, ATTN_DIM_SIZE) * WEIGHT_SCALE


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

    # prev:
    # def attention (self, x_seq):
    #    Q = np.dot(x_seq, self.Wq)
    #    K = np.dot(x_seq, self.Wk)
    #    V = np.dot(x_seq, self.Wv)
        
    #   self.Q= Q
    #    self.K= K
    #    self.V= V

    #    scale = np.sqrt (head_dim_size)
    #    scores = np.dot(Q, K.T) / scale # (N, N)

    #    scores -=np.max(scores, axis=1, keepdims=True)
    #    weights = np.exp(scores)
    #    weights/= weights.sum(axis=1, keepdims=True)
    #    self.attn_weights= weights

    #    out= np.dot(weights, V) # (N, ATTN_DIM)
    #    return out, weights
    
    # curr:
    # for each head:
    #   compute q,k,v
    #   compute attention output
    # collect all heads
    # concantenate
    # apply Wo
    def attention(self, x_seq):
        head_outputs= []
        all_weights= []
        all_Q, all_K, all_V = [], [], []

        for h in range (self.num_heads):
            Wq = self.Wq[h]
            Wk = self.Wk[h]
            Wv = self.Wv[h]

            # compute Q,K,V  for this seed
            Q= x_seq @Wq
            K= x_seq @Wk
            V= x_seq @Wv

            # save as all logs for backprop
            all_Q.append(Q)
            all_K.append(K)
            all_V.append(V)

            # a much more stable softmaxx
            scale = np.sqrt(self.head_dim_size)
            scores = (Q @K.T) /scale
            scores -= np.max (scores, axis=1, keepdims=True)
            weights = np.exp(scores)
            weights/= weights.sum (axis=1, keepdims=True)

            out= weights @ V

            # store results
            head_outputs.append(out)
            all_weights.append(weights)

        # concatenate all heads
        concat= np.concatenate(head_outputs, axis=1)

        # final projection
        out = concat @ self.Wo

        # save for backprop
        self.head_outputs = head_outputs
        self.attn_weights= all_weights
        self.all_Q= all_Q
        self.all_K= all_K
        self.all_V= all_V

        return out, all_weights
    
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

        # reshape to sequence
        x_seq= x.reshape(self.n, EMBED_DIM_SIZE)
        self.x_seq= x_seq

        # attention
        attn_out, self.attn_weights = self.attention(x_seq)
        attn_out = attn_out + x_seq

        # flatten back
        x= attn_out.reshape (1,-1) # (1, ATTN_DIM_SIZE*N)

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

        dw1 = np.dot(self.x.T, dz1)             # shape: (ATTN_DIM_SIZE * N, hidden_size)
        db1 = dz1                               # shape: (1, hidden_size)
        
        # nudge every weight
        # new_weight = old_weight - learning_rate × gradient
        self.w2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2
        self.w1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1

        # removed old embedding block
        # attention backprop implementation
        dx = np.dot (dz1, self.w1.T) # (1, ATTN_DIM_SIZE *N)

        # reshape to (N, ATTN_DIM_SIZE)
        d_attn_out = dx.reshape (self.n, -1)

        # gradients kinda explode lowkirkenuinely at around 200 epochs
        # try gradient clipping to prevent it and loss actually goes down
        CLIP = 1.0
        d_attn_out = np.clip(d_attn_out, -CLIP, CLIP)

        # Multi-head attention backward
        # prev: attention output was flattened into (1, ATTN_DIM_SIZE*N)
        # curr: I restored it back into sequence form so each token has its vector again

        # backprop through Wo
        # prev: attention directly output (N, ATTN_DIM_SIZE)
        # curr: we have concat_heads to Wo to final attention output

        # reconstruct what was fed into Wo during forward
        concat = np.concatenate(self.head_outputs, axis=1) # (N, ATTN_DIM_SIZE)
        
        # gradient wrt Wo usually linear layer rule of X^T @ dY
        dWo = concat.T @ d_attn_out

        # gradient flowing back before Wo
        d_concat= d_attn_out @ self.Wo.T
        self.Wo-= learning_rate * dWo

        # each head contributes to gradient back to same input sequence
        # so we sum all head contributions
        d_x_seq= np.zeros_like(self.x_seq)

        # loop over heads
        # prev: one attention pipeline
        # curr: same pipeline repeated per head then summed
        for h in range (self.num_heads):

            # slice this head's portion
            # because forward did : concat (head1, head 2, so on...)
            start_slice=h * self.head_dim_size
            end_slice = (h+1) * self.head_dim_size
            d_out = d_concat[:, start_slice:end_slice]

            # retreive stored forward value for curr head
            Q = self.all_Q[h]
            K = self.all_K[h]
            V = self.all_V[h]
            weights = self.attn_weights[h]
            
            # out= weights @ V
            # basically same as prev single head, now just per head
            # absolute cinema
            dV= weights.T @ d_out
            dih_weights = d_out @ V.T

            # softmaxx backward
            # scaled by head_dim instead of full ATTN_DIM
            scale= np.sqrt (self.head_dim_size)
            dih_scores = weights * (dih_weights- (dih_weights * weights).sum(axis=1, keepdims=True))
            dih_scores/= scale

            # scores = Q @ K.T
            dQ = dih_scores @ K
            dK = dih_scores.T @ Q

            # gradients for projection weights
            # each head has own Wq, Wk, Wv
            dWv = self.x_seq.T @ dV
            dWq = self.x_seq.T @ dQ
            dWk = self.x_seq.T @ dK

            # backprop for x_seq
            # each head contribute t same input then accumulate 
            d_x_seq +=(
                dV @ self.Wv[h].T +
                dQ @ self.Wq[h].T +
                dK @ self.Wk[h].T 
            )

            # update weights per head
            # each head updates its own parameters now
            self.Wv[h]-= learning_rate* dWv
            self.Wq[h]-= learning_rate* dWq
            self.Wk[h]-= learning_rate* dWk

        # residual gradient
        # skip path from attn_out + x_seq in forward
        d_x_seq += d_attn_out

        # update embeddings
        for i, idx in enumerate(self.input_indices):
            self.e[idx]-= learning_rate * d_x_seq[i]

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
    