# main.py
import sys
import os
import numpy as np
from data import build_vocab, get_ngram_pairs, sentences
from network import net
from generate import generate
from constants import N, HIDDEN_SIZE, EPOCHS, LEARNING_RATE, PRINT_EVERY, MODEL_PATH, GEN_LENGTH

VIZ = "--viz" in sys.argv

word_to_idx, idx_to_word = build_vocab(sentences)
vocab_size = len(word_to_idx)

# train if no saved model
if not os.path.exists(MODEL_PATH):
    print("no saved model found. training...\n")

    network = net(vocab_size=vocab_size, hidden_size=HIDDEN_SIZE, n=N)
    pairs   = get_ngram_pairs(sentences, word_to_idx, N)

    loss_log  = []
    epoch_log = []

    for epoch in range(EPOCHS):
        total_loss = 0
        for inp, tgt in pairs:
            probs = network.forward(inp)
            total_loss += network.loss(probs, tgt)
            network.backward(tgt, learning_rate=LEARNING_RATE)

        avg_loss = total_loss / len(pairs)

        if epoch % PRINT_EVERY == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:>5} | loss: {avg_loss:.4f}")
            loss_log.append(avg_loss)
            epoch_log.append(epoch)

    np.savez(MODEL_PATH,
             w1=network.w1, w2=network.w2,
             b1=network.b1, b2=network.b2,
             loss_log=np.array(loss_log),
             epoch_log=np.array(epoch_log))

    print("\ntraining complete. model saved.\n")

# load model.npz
network = net(vocab_size=vocab_size, hidden_size=HIDDEN_SIZE, n=N)
saved   = np.load(MODEL_PATH)
network.w1 = saved["w1"]
network.w2 = saved["w2"]
network.b1 = saved["b1"]
network.b2 = saved["b2"]

loss_history  = list(saved["loss_log"])  if "loss_log"  in saved else []
epoch_history = list(saved["epoch_log"]) if "epoch_log" in saved else []

if VIZ:
    from visualize import show_generation
    print("visualization mode on. a plot will appear after each generation.\n")

while True:
    seed = input(f"\nenter seed ({N} words) or 'quit': ").strip()
    if seed == "quit":
        break

    words = seed.split()

    if len(words) != N:
        print(f"enter exactly {N} words")
        continue

    unknown = [w for w in words if w not in word_to_idx]
    if unknown:
        print(f"unknown words: {unknown}")
        continue

    result = generate(seed, GEN_LENGTH, network, word_to_idx, idx_to_word, N)
    print(f"\n{result}")

    if VIZ:
        generated = result.split()[N:]   # keep only predicted words
        show_generation(
            seed_words      = words,
            generated_words = generated,
            network         = network,
            word_to_idx     = word_to_idx,
            idx_to_word     = idx_to_word,
            loss_history    = loss_history,
            epoch_history   = epoch_history,
            n               = N,
            epochs          = EPOCHS,
        )