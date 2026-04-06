#rot'nn
a predictive neural network from scratch using raw numpy.
No pytorch or tensorflow.

Trained on brainrot-> generates brainrot

Eg.
-input: "I just spent"
-output: "I just spent 18 million robux on this shirt say bro wallahi sigma"

The output is genuinely dumb, but I purposefully built this from scratch
so I can learn how language models actually work under the hood.
This is an experimental project, and I will be doing major changes as I continue
and deepen my learning of neural networks

##how it works:
a sentence is split into each word. Each word gets conveted to a number.
The network takes N words at a time, tries to guess the next one, checks
how wrong it was, then nudges its weights to be less wrong. Repeat 5000 times
(or how many times you want it is configurable via constants.py), that's it.

##architecture (if you want the nerdy technical explanation):
-input: N one-hot vectors concatenated -> shape (1, vocab_size × N)
-hidden layer: linear transformation + ReLU activation
-output layer: linear transformation + softmax -> probability 
distribution over vocab
-loss: cross-entropy
-optimization: weight = weight - learning_rate × gradient via backpropagation

##installation
clone the repo:
```bash
git clone https://github.com/SelwynLatog/rot-nn.git
cd rot-nn
```

install dependencies:
```bash
pip install numpy matplotlib
```

##how to run
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
#windows
python main.py --viz
#linux/mac
rm model.npz
```
to see the visualization of the network

###NOTE:
when you want to rerun/retrain. Simply delete 'model.npz' and rerun main
```bash:
del model.npz
python main.py --viz
```

