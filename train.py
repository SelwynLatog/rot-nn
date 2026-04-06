# train.py
# loops through training pairs -> forward -> loss -> backward -> updates weights
# prints loss over time and saves trained model as model.npz
import numpy as np
from data import build_vocab, get_ngram_pairs, sentences
from network import net
from constants import N, EPOCHS, LEARNING_RATE, HIDDEN_SIZE, PRINT_EVERY
# setup
word_to_idx, idx_to_word = build_vocab(sentences)
vocab_size = len(word_to_idx)

network = net(vocab_size= vocab_size, hidden_size=HIDDEN_SIZE, n=N)
pairs = get_ngram_pairs(sentences, word_to_idx,n=N)

epochs = EPOCHS # how many passes through all training pairs
learning_rate = LEARNING_RATE
print_every = PRINT_EVERY

print (f"training on {len(pairs)} pairs for {epochs} epochs")
print (f"vocab size :{vocab_size}")

# training loop
for epoch in range (epochs):
    total_loss=0

    # run every training pair once per epoch
    for inp, tgt in pairs:
        # forward pass
        probs = network.forward(inp)
        # measure loss
        total_loss+= network.loss(probs, tgt)
        # backward pass
        network.backward(tgt, learning_rate= learning_rate)
    
    # avg loss across all pairs this epoch
    avg_loss = total_loss/len(pairs)
    if epoch % print_every == 0 or epoch == epochs -1:
        print(f"epoch {epoch:>5} | loss: {avg_loss:.4f}")
print(f"training complete")

print(f"\n predictions post training:")
test_words = [p[0] for p in pairs]

seen = set()
for inp in test_words:
    inp_tuple = tuple(inp)
    if inp_tuple in seen:
        continue
    seen.add(inp_tuple)

    probs = network.forward(inp)
    predicted = np.argmax(probs)
    confidence = probs[0, predicted]

    input_words = [idx_to_word[i] for i in inp]

    print(f"  {input_words} -> '{idx_to_word[predicted]}'  ({confidence:.0%} confident)")

np.savez("model.npz", w1=network.w1, w2=network.w2, b1=network.b1, b2=network.b2)
print("model saved.")