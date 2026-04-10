# data.py
# simply reads and splits
# every word in the sentences and splits based on NGram size

# add/delete/ modify as many sentences as you want
sentences = [
    # uncomment/comment this data set for the original brainrot output
    # hard to tell improvements with added single-head attention layer
    # due to the dataset thenmselves being incomprehensible in the first place
    # no shit sherlock

    # ishowspeed 10/10 imdb EARLY STREAM
    #"please speed i need this",
    #"my momma's kinda homeless",
    #"my momma's has no property",
    #"speed is laughing because his momma's homeless",
    #"it is disrespectful that speed is laughing",
    #"speed laughs because his momma was kinda homeless",
    #"why you laughing bruh that is disrespectful bruh",
    #"it is not funny that his momma is homeless bruh",

    # ishowspeed roblox incident group
    #"i just spent 200k dollars on a shirt",
    #"speed spent money on roblox"
    #"did i just spent 18 million robox on this shirt",
    #"say you swear to god I just spent",
    #"say wallahi bro say you swear",

    # 3 netherite ingots incident
    #"we have 24 gold ninjas more",
    #"34 gold ninjas ingots",
    #"3 netherite ingots oh fuck yes sir",
    #"yes sir we have 3 ninjarite ingots",
    
    # common brain rot
    #"you're so skibidi marlon duke dennis",
    #"what the sigma duke dennis",
    #"rizz the huzz marlon duke dennis"

    
    # here is a much more simple data set so data output is much more comprehensible
    # I am sick of the fucking brain rot
    # this dataset gives a much more clear demo on how the attention layer works
    # it groups related words eg:
    # 'ate food' -> 'hungry'
    # 'drank water' -> 'thirsty'
    # 'tired' -> 'sleepy'

    "i ate food because i was hungry",
    "i drank water because i was thirsty",
    "i slept because i was tired",
    "i ran because i was late",
    "he ate food because he was hungry",
    "she drank water because she was thirsty",
    "they slept because they were tired",
    "the cat ate food because it was hungry",
    "the dog ran because it was excited",
    "i studied because i had a test",
    "i practiced because i wanted to improve",
    "he trained because he wanted to win"

]

def build_vocab(sentences):
    # reads every word and assigns indexes in alphabetical order
    # words are split by spaces
    # "say wallahi" -> ["say", "wallahi"]
    words = set()
    for sentence in sentences:
        for word in sentence.split():
            words.add(word)
    words = sorted(words)
    word_to_idx = {word: idx for idx, word in enumerate(words)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return word_to_idx, idx_to_word

def get_training_pairs(sentences, word_to_idx):
    pairs = []
    for sentence in sentences:
        words = sentence.split()
        for i in range(len(words) - 1):
            input_word  = word_to_idx[words[i]]
            target_word = word_to_idx[words[i + 1]]
            pairs.append((input_word, target_word))
    return pairs

def get_ngram_pairs(sentences, word_to_idx, n):
    pairs = []
    
    for sentence in sentences:
        words = sentence.split()
        
        for i in range(len(words) - n):
            input_words = words[i: i+n] # grab n words
            target_word = words[i + n]  # grab the answer
            
            input_indices = [word_to_idx[w] for w in input_words]
            target_index  = word_to_idx[target_word]
            
            pairs.append((input_indices, target_index))
    
    return pairs

if __name__ == "__main__":
    word_to_idx, idx_to_word = build_vocab(sentences)
    pairs = get_training_pairs(sentences, word_to_idx)
    print("vocabulary")
    for word, idx in word_to_idx.items():
        print(f"  {idx:2d}  ->  {word}")
    print(f"\ntotal unique words: {len(word_to_idx)}")
    for inp, tgt in pairs:
        print(f"  '{idx_to_word[inp]}' -> '{idx_to_word[tgt]}'")
    print(f"\ntotal training pairs: {len(pairs)}")
    print("\nngram pairs (n=2):")
    ngram_pairs = get_ngram_pairs(sentences, word_to_idx, 2)
    for inp, tgt in ngram_pairs[:5]:
        input_words = [idx_to_word[i] for i in inp]
        print(f"  {input_words} -> '{idx_to_word[tgt]}'")