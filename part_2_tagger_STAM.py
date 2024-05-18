import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize


# Create tagger
text = "We will play football tomorrow"
tokens = nltk.word_tokenize(text)
tagged_tokens = nltk.pos_tag(tokens)

def_text_tagger = nltk.DefaultTagger('PRP')
def_tagged_tokens = def_text_tagger.tag(tokens)

# Uniform taggers with backoff using Brown Corpus and check accuracy
train_sents = brown.tagged_sents(categories='news')
uni_tagger_one = nltk.UnigramTagger(train_sents, backoff=def_text_tagger)
ac1 = uni_tagger_one.evaluate(train_sents)
print(ac1)

train_sents = brown.tagged_sents(categories='adventure')
uni_tagger_two = nltk.UnigramTagger(train_sents, backoff=uni_tagger_one)
ac2 = uni_tagger_two.evaluate(train_sents)
print(ac2)

train_sents = brown.tagged_sents(categories='humor')
uni_tagger_three = nltk.UnigramTagger(train_sents, backoff=uni_tagger_two)
ac3 = uni_tagger_three.evaluate(train_sents)
print(ac3)

# N-gram taggers with backoff using Brown Corpus
train_sents = brown.tagged_sents(categories='mystery')
bigram_tagger_one = nltk.BigramTagger(train_sents, backoff=def_text_tagger)
acn1 = bigram_tagger_one.evaluate(train_sents)
print(acn1)

train_sents = brown.tagged_sents(categories='hobbies')
bigram_tagger_two = nltk.BigramTagger(train_sents, backoff=bigram_tagger_one)
acn2 = bigram_tagger_two.evaluate(train_sents)
print(acn2)

train_sents = brown.tagged_sents(categories='learned')
bigram_tagger_three = nltk.BigramTagger(train_sents, backoff=bigram_tagger_two)
acn3 = bigram_tagger_three.evaluate(train_sents)
print(acn3)

# Combination taggers
train_sents_size_1 = train_sents[:2000]
test_sents_size_1 = train_sents[2000:]
def_con_1 = nltk.DefaultTagger('RB')
uni_con_1 = nltk.UnigramTagger(train_sents_size_1, backoff=def_con_1)
big_con_1 = nltk.BigramTagger(train_sents_size_1, backoff=uni_con_1)
ac1_con_1 = big_con_1.evaluate(test_sents_size_1)
print(ac1_con_1)

train_sents_size_2 = train_sents[:500]
test_sents_size_2 = train_sents[500:]
def_con_2 = nltk.DefaultTagger('IN')
big_con_2 = nltk.BigramTagger(train_sents_size_2, backoff=def_con_2)
uni_con_2 = nltk.UnigramTagger(train_sents_size_2, backoff=big_con_2)
ac2_con_2 = uni_con_2.evaluate(test_sents_size_2)
print(ac2_con_2)

train_sents_size_3 = train_sents[:1500]
test_sents_size_3 = train_sents[1500:]
def_con_3 = nltk.DefaultTagger('JJ')
uni_con_3 = nltk.UnigramTagger(train_sents_size_3, backoff=def_con_3)
big_con_3 = nltk.BigramTagger(train_sents_size_3, backoff=uni_con_3)
ac3_con_3 = big_con_3.evaluate(test_sents_size_3)
print(ac3_con_3)

