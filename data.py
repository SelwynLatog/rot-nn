# data.py
# simply reads and splits
# every word in the sentences and splits based on NGram size

# add/delete/ modify as many sentences as you want
sentences = [
    "please speed i need this",
    "my momma's kinda homeless",
    "say you swear to god",
    "say wallahi bro say wallahi",
    "i just spent 200k dollars on a shirt",
    "did i just spent 18 million robox on this shirt",
    "there's a negev on site on b",
    "give him a negev give him a negev",
    "woah cancel him",
    "the reddit will hear about this",
    "you're so skibidi",
    "what the sigma",
    "rizz the huzz",
    "clav just got frame mogged by the asu frat leader",
    "frat leader mogs looksmaxxing sigma",
    "subhuman tries to mog rizzler",
    "netflix forgets steel ball run",
    "lebron james balls on my face",
    "ai fruit love island is cinema",
    "socrates ragebaits skeleton mogged",
    "i think therefore i goon",
    "six seven compilation",
    "kai cenat performative",
    "marlon mogging charlie kirk",
    "terry davis the best programmer",
    "programming sucks",
    "linus torvalds is so petty",
    "windows can kiss my ass",
    "your ip has been leaked",
    "gentoo users have type 2 diabetes",
    "i am not a racist",
    "fartinhaler ranks up to mythic 2",
    "the double champ does what the fuck he wants",
    "have you ever been fingered by an mma fighter before",
    "its like watching the mona lisa being painted"
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