# constants.py
# tweak these to experiment network behavior

# Ngram context window (word input size) 
# how many words the network sees before guessing

# The current model is a multi head attention mini transformer
# Trigram produce garbage output despite beautiful loss drop curve due to overfitting
# Update:
# Expand sentences in 'data.py' and change to a Pentagram context window
# Update update: its late. I dont know anymore. I'll revert back to Trigram. This has the best output
# N5 JUST FUCKING EXPLODES CONTEXT WIN TOO LARGE AND OUTPUT GETS GARBAGE
# N3 is the best
N=3

# Embedding matrix size
EMBED_DIM_SIZE= 16

# Attension matrix size
ATTN_DIM_SIZE = 16

# neurons in the hidden layer
# more equals less monkey but slower
HIDDEN_SIZE= 64

# Multi-head attention layer implementation
NUM_HEADS= 4

# how many full passes through the training data
# more sentences you add in data.py means the more epochs
# you will likely need

# Insights during testing: 
# I found to be very interesting is, the original N-Gram memorizor just needed around 5k epochs for min loss
# converted to single head attention & embeddings and it took much longer at around 15k epochs

# toggle epoch at around 5000 - 50000 depending on how fast or slow your learning rate is
# but hopefully (say wallahi bro), max loss would be lower while keeping loss drop slow and consistent
# toggle epoch depending on how 
EPOCHS = 25000

# one cause of repetitive garbage outputs is training near 100% confidence
# which causes repetitive answers no matter how we scale up k filtering & temperature
# basically I realized that

# if its too low for the context window = it becomes a parrot
# if its just right = a learning 1 year old toddler which is what im trying to achieve
# if its too high = down syndorme
MIN_LOSS= 0.5

# how big the correction is with every step. Too High = Unstable. Too Low= Slow
# update: set a lot lower now due to the new attention weights
# lower end is incredibly slow (it took a whole night of training i am not even joking)
# but loss drop was beautiful and stable
# toggle at around 0.5-0.01
LEARNING_RATE= 0.05

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
TEMPERATURE = 1.4

# saved path for the trained weights
MODEL_PATH = "model.npz"