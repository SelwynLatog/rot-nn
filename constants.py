# constants.py
# tweak these to experiment network behavior

# Ngram context window (word input size) 
# how many words the network sees before guessing

# The current model is a multi head attention mini transformer
# Trigram produce garbage output despite beautiful loss drop curve due to overfitting
# Update:
# Expand sentences in 'data.py' and change to a Pentagram context window
N=5

# Embedding matrix size
EMBED_DIM_SIZE= 16

# Attension matrix size
ATTN_DIM_SIZE = 16

# neurons in the hidden layer
# more equals less monkey but slower
HIDDEN_SIZE= 16

# Multi-head attention layer implementation
NUM_HEADS= 4

# how many full passes through the training data
# more sentences you add in data.py means the more epochs
# you will likely need

# Insights during testing: 
# I found to be very interesting is, the original N-Gram memorizor just needed around 5k epochs for min loss
# converted to single head attention & embeddings and it took much longer at around 15k epochs

# converted to a multi head attention and model learns much faster and just needed approx. 3k epochs for min loss without decay
# but that causes overfitting on low num of pairs with Trigram
# this produces garbage output despite high confidence

# solution would be to try and make decay way less aggressive in 'main.py'
# let epoch cycle longer same as single head which had the best output
# but hopefully (say wallahi bro), max loss would be lower while keeping loss drop slow and consistent
# tradeoff would be it takes like 3-4 minutes training each run
EPOCHS = 15000

# how big the correction is with every step. Too High = Unstable. Too Low= Slow
# update: set a lot lower now due to the new attention weights
# prev 0.01 caused it to be unstable
LEARNING_RATE= 0.001

# scales initial random weights. Too high = unstable gradients. Too low = slow start
WEIGHT_SCALE = 0.01

# how often to print loss during training
# for visualization unless you have a decent pc you might want to stay
# within the higher range
PRINT_EVERY= 500

# number of words to generate after the seed
GEN_LENGTH = 10

# exploratory var pre softmax
# lower= consecutive & robotic, higher = chaotic & exploratory 
TEMPERATURE = 1

# saved path for the trained weights
MODEL_PATH = "model.npz"