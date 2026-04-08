from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer as wnl
from nltk import pos_tag
from glob import glob
import nltk
from nltk.corpus import wordnet
nltk.download("punkt")
nltk.download("stopwords")

class FreshException(Exception):
    pass

def read_file(topic, language):
    try:
        path = f"{topic}/{language}_{topic}"
        with open(f'data/{path}.txt', 'rb') as f:
            txt = f.read().decode().strip().lower()

    except:
        raise Exception("Please put in a valid topic and language. Make sure you are putting in the full topic name.")
    return txt


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
    

def tokenize(text, lang):
    lemma = wnl()
    stop_words = set(stopwords.words(lang))

    tokens = word_tokenize(text)
    tokens = pos_tag(tokens, tagset='universal')
    
    for i, pair in enumerate(tokens):
        word, tag = pair
        if word not in stop_words and word not in string.punctuation:
            pos = get_wordnet_pos(tag)
            if pos:
                tokens[i] = lemma.lemmatize(word, pos=pos).lower()
            else:
                tokens[i] = lemma.lemmatize(word).lower()
    return tokens

def word_2_vec(tokens, path, vs=128, context=5):
    print('Inititalizing new model...')
    model = Word2Vec(sentences=[tokens], vector_size=vs, window=context, min_count=1, workers=4)

    # write to path  
    model.save(path)
    return model.wv