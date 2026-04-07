# constants.py
# tweak these to experiment network behavior

# Ngram context window (word input size) 
# how many words the network sees before guessing
N=3

# Embedding matrix size
# Will change it from a one_hot vector to an embed via 'network.py'
EMBED_DIM_SIZE= 16

# neurons in the hidden layer
# more equals less monkey but slower
HIDDEN_SIZE= 16

# how many full passes through the training data
# more sentences you add in data.py means the more epochs
# you will likely need
EPOCHS = 5000

# how big the correction is with every step. Too High = Unstable. Too Low= Slow
LEARNING_RATE= 0.01

# scales initial random weights. Too high = unstable gradients. Too low = slow start
WEIGHT_SCALE = 0.01

# how often to print loss during training
# for visualization unless you have a decent pc you might want to stay
# within the higher range
PRINT_EVERY= 500

# number of words to generate after the seed
GEN_LENGTH = 10

# saved path for the trained weights
MODEL_PATH = "model.npz"