# rot'nn

## v1
a simple brainrot predictive neural network from scratch using raw numpy.
No pytorch or tensorflow.
Trained on brainrot-> generates brainrot

This kinda just started as a joke. "Let me build a neural network from scratch but its brainrot haha"
I got obsessed with it, it became a mini transformer

## v2
a mini transformer. It learned embeddings, sinusoidal positional encoding, 
multi-head self-attention, and residual connections.
Essentially it went from a parrot to an artificial one year old toddler..somewhat.


### v1 sample output
<img width="602" height="499" alt="rotnn1" src="https://github.com/user-attachments/assets/1d501be7-7de5-4cd2-a226-7445b212e062" />

<img width="826" height="529" alt="rotnn2" src="https://github.com/user-attachments/assets/6a139798-1f07-48e3-9d67-6d45d903f7d0" />

<img width="277" height="182" alt="image" src="https://github.com/user-attachments/assets/3ea2e9d6-4f62-4e99-9047-9e3f2164762a" />

<img width="940" height="584" alt="rotnn4" src="https://github.com/user-attachments/assets/c1ff770a-9cdb-4eeb-80e9-035c4653694a" />

<img width="931" height="572" alt="rotnn5" src="https://github.com/user-attachments/assets/8f941925-a79f-425d-9caa-66da36a3c81a" />



Eg.
-input: "I just spent"
-output: "I just spent 18 million robux on this shirt say bro wallahi sigma"

v1:
The output is genuinely dumb, but I purposefully built this from scratch
so I can learn how language models actually work under the hood.
This is an experimental project, and I will be doing major changes as I continue
and deepen my learning of neural networks

v2:
the output is still broken english. but it's *different* broken english.
it pulls real fragments from training. it has syntactic flow.
it's not random noise anymore. it's a 1 year old that learned some words kinda.

## how it works:
a sentence is split into each word. Each word gets converted to a number.
The network takes N words at a time, tries to guess the next one, checks
how wrong it was, then nudges its weights to be less wrong. Repeat 5000 times
(or how many times you want it is configurable via constants.py), that's it.

## v1 architecture (if you want the nerdy technical explanation):
-input: N one-hot vectors concatenated -> shape (1, vocab_size × N)
-hidden layer: linear transformation + ReLU activation
-output layer: linear transformation + softmax -> probability 
distribution over vocab
-loss: cross-entropy
-optimization: weight = weight - learning_rate × gradient via backpropagation

## v2 architecture:
-same numpy. completely different model.
-swapped one-hot for learned embeddings + sinusoidal positional encoding
-added multi-head attention (4 heads) with full backprop
-trained on 100+ normal english sentences. ~24k epochs. took a whole night
-loss bottomed out around sub 1 range

### v1 here's a badly drawn diagram
<img width="1123" height="694" alt="image" src="https://github.com/user-attachments/assets/0c3836c6-8200-4380-b76d-97599f1355e8" />

## installation
clone the repo:
```bash
git clone https://github.com/SelwynLatog/rot-nn.git
cd rot-nn
```

install dependencies:
```bash
pip install numpy matplotlib
```

## how to run
-run main:
```bash
python main.py
```
-it will automatically train and generate 'model.npz'
```bash
no saved model found. training...

epoch     0 | loss: 4.9559
epoch   500 | loss: 0.0590
epoch  1000 | loss: 0.0285
epoch  1500 | loss: 0.0239
epoch  2000 | loss: 0.0219
epoch  2500 | loss: 0.0207
epoch  3000 | loss: 0.0199
epoch  3500 | loss: 0.0193
epoch  4000 | loss: 0.0189
epoch  4500 | loss: 0.0185
epoch  4999 | loss: 0.0183

training complete. model saved.


enter seed (3 words) or 'quit':
```
-seed is N configurable via 'constants.py'

```bash
enter seed (3 words) or 'quit': please speed i

please speed i need this 18 million robox on this shirt wallahi run

enter seed (3 words) or 'quit': what the sigma

what the sigma he wants homeless ball run sigma homeless what homeless fuck

enter seed (3 words) or 'quit': clav just got

clav just got frame mogged by the asu frat leader island looksmaxxing sigma

enter seed (3 words) or 'quit': lebron james balls

lebron james balls on my face leaked homeless ball run face mma frame

enter seed (3 words) or 'quit': quit
```
Optionally you can:
```bash
python main.py --viz
```
to see the visualization of the network

### NOTE:
when you want to rerun/retrain. Simply delete 'model.npz' and rerun main
```bash:
# windows
del model.npz
# linux/mac
rm model.npz
python main.py --viz
```
