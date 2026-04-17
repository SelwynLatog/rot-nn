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

    # brainrot phrases commented out to showcase model more clearly.
    # (I am sick of the fucking brain rot)
    
    # NOTE: The model in itself is tiny. One attention layer, no layer norm,
    # no dropout, no deep stacking.
    # It can't learn nuanced long-range context,
    # only strong local co-occurrence patterns.
    # so if the dataset is pre dominant with pronounds,
    # the model will likely have garbage pronoun repeating output.
    # the model itself has learned. Attention, embeddings, as loss curve goes down
    # at a very stable rate. At the moment it's close to an artificial 1 year old
    # which is the goal at the moment (i do not want to go insane)

    # this dataset gives a good output of the model's capabilities
    # tailored specifically so it can actually form somewhat coherent sentences
    # this has been a very fun project

    "the dog barked because it was scared",
    "the cat slept near the warm window",
    "the bird flew away from the loud noise",
    "the fish swam deep into the cold water",
    "the dog ran fast across the open field",
    "the cat knocked the cup off the table",
    "the bird sang early in the morning",
    "the fish jumped out of the water suddenly",
    "the dog ate quickly because it was hungry",
    "the cat hid under the bed during the storm",

    "the rain fell hard on the empty street",
    "the wind knocked the leaves off the trees",
    "the sun set slowly behind the hills",
    "the clouds covered the sky before the storm",
    "the snow melted when the sun came out",
    "the thunder was loud and the dog hid",
    "the rain stopped and the sky turned orange",
    "the wind blew cold across the open field",
    "the fog made it hard to see the road",
    "the storm passed and everything was quiet",

    "she laughed when the cat fell off the chair",
    "he cried because he lost his favorite book",
    "she smiled when she heard the good news",
    "he frowned when the food was too cold",
    "she jumped when she heard the loud noise",
    "he paused before he answered the question",
    "she whispered because the baby was sleeping",
    "he shouted because no one was listening",
    "she stopped walking when she saw the dog",
    "he looked up when the lights went out",

    "the coffee was too hot to drink",
    "the bread smelled fresh out of the oven",
    "the soup was warm and tasted like home",
    "the cake fell apart before anyone ate it",
    "the apple rolled off the table onto the floor",
    "the milk spilled when the cup tipped over",
    "the rice burned because the heat was too high",
    "the tea cooled down before she could drink it",
    "the egg cracked when it hit the floor",
    "the fruit was sweet and easy to eat",

    "he studied late because the exam was tomorrow",
    "she forgot her bag and went back home",
    "he missed the bus and had to walk",
    "she finished her work before the sun went down",
    "he lost his keys and could not get in",
    "she read the letter twice before she understood",
    "he packed his bag the night before",
    "she woke up late and skipped breakfast",
    "he arrived early and waited outside",
    "she left without saying anything to anyone",

    "the old man sat quietly by the river",
    "the little girl dropped her ice cream on the ground",
    "the boy climbed the tree to get a better view",
    "the woman ran to catch the bus before it left",
    "the child cried when the balloon flew away",
    "the man opened the window to let in fresh air",
    "the girl read under the tree all afternoon",
    "the boy tripped and dropped everything he was carrying",
    "the woman cooked dinner while the rain fell outside",
    "the man fell asleep before the movie ended",

    "the light turned off when everyone left the room",
    "the door opened slowly and no one was there",
    "the phone rang but no one picked it up",
    "the alarm went off before the sun came up",
    "the clock stopped working in the middle of the night",
    "the candle flickered when the wind came through",
    "the window fogged up because of the cold outside",
    "the floor creaked when he walked across it",
    "the fan spun slowly in the hot afternoon",
    "the screen went dark when the battery died",

    "she felt nervous before she walked on stage",
    "he felt relieved when the test was finally over",
    "she felt proud when she finished the project",
    "he felt confused when the instructions changed",
    "she felt tired but kept going anyway",
    "he felt excited but tried not to show it",
    "she felt calm after she talked to her friend",
    "he felt lonely even when the room was full",
    "she felt grateful when someone helped her",
    "he felt embarrassed when he said the wrong name",

    "the puppy tripped over its own feet running",
    "the kitten stared at the wall for no reason",
    "the rabbit stopped and looked around before moving",
    "the bird tapped the window three times then left",
    "the squirrel buried something and forgot where it went",
    "the duck walked in a straight line across the road",
    "the dog stared at the door waiting for someone",
    "the cat sat in the box even though it was too small",
    "the hamster ran on the wheel all night long",
    "the parrot repeated the last word it heard",

    "after the rain the street smelled clean and fresh",
    "before the storm the birds all went quiet",
    "when the music stopped everyone looked around",
    "after she left the room felt very empty",
    "when he arrived everything changed immediately",
    "before she answered she took a deep breath",
    "when the food arrived they forgot what they were saying",
    "after the exam he slept for twelve hours straight",
    "when the lights came back on everyone cheered",
    "before the game started they were all very quiet",

    "he tried to explain but the words came out wrong",
    "she wanted to help but did not know how",
    "he meant to call but forgot until it was late",
    "she planned to leave early but stayed until the end",
    "he tried to be quiet but knocked something over",
    "she wanted to sleep but her mind would not stop",
    "he forgot what he was saying halfway through",
    "she almost left but turned back at the door",
    "he started the work then stopped and stared outside",
    "she reached for the cup and knocked it over instead",

    "the street was empty except for one old bicycle",
    "the park bench was wet from the rain last night",
    "the library was silent except for turning pages",
    "the kitchen smelled like something was burning slowly",
    "the hallway was dark and the light was broken",
    "the garden looked different after the storm passed through",
    "the road curved sharply before the old bridge",
    "the market was loud and full of people",
    "the rooftop was quiet and the view was wide",
    "the staircase creaked every time someone walked up",

    "he ate slowly because he was not really hungry",
    "she drank water even though she wanted something sweet",
    "he walked fast even though he had time",
    "she spoke softly even though no one was sleeping",
    "he smiled even though the day had been hard",
    "she kept going even though she was very tired",
    "he stayed quiet even though he had something to say",
    "she waited patiently even though it was taking too long",
    "he laughed even though the joke was not that funny",
    "she stayed calm even though everything was going wrong",

    "the old clock on the wall ticked very loudly",
    "the small plant on the desk needed more water",
    "the book on the floor had lost its cover",
    "the bag by the door was too heavy to carry",
    "the note on the table said gone for a walk",
    "the chair by the window was her favorite spot",
    "the key under the mat was not there anymore",
    "the cup on the shelf was chipped on one side",
    "the light above the table buzzed when it was on",
    "the map on the wall had marks all over it",

    "she hummed while she washed the dishes alone",
    "he whistled while he walked down the empty road",
    "she drew circles on the paper while she thought",
    "he tapped his foot while he waited in line",
    "she stared out the window while the meeting went on",
    "he chewed slowly while he read the long message",
    "she nodded while he talked even though she disagreed",
    "he blinked slowly while the screen loaded",
    "she twisted her ring while she thought about what to say",
    "he leaned back and looked at the ceiling for a while",

    "the first bite was good but the second was better",
    "the first day was hard but it got easier after",
    "the first word was wrong so she started over",
    "the last piece of bread was saved for later",
    "the last person to leave turned off all the lights",
    "the last message he sent was never answered",
    "the only window in the room faced a blank wall",
    "the only sound was the rain hitting the roof",
    "the only chair left was broken on one side",
    "the first step was small but it still counted",

    "something moved in the corner of the room",
    "nothing happened for a long time then everything did",
    "someone left the door open and the cat got out",
    "nobody noticed when the picture fell off the wall",
    "everything smelled different after the long rain",
    "nothing was where she left it the night before",
    "someone had moved the chair to the other side",
    "everybody clapped except for one person in the back",
    "somewhere outside a dog was barking at nothing",
    "something about the morning felt different today",

    "he did not remember falling asleep on the couch",
    "she did not notice the mud on her shoes",
    "he did not hear the door close behind him",
    "she did not expect the answer to be that simple",
    "he did not finish the sentence before walking away",
    "she did not look back when she left the room",
    "he did not say much but his face said everything",
    "she did not mind waiting because the view was nice",
    "he did not realize how tired he was until he sat down",
    "she did not know the song but hummed along anyway"
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