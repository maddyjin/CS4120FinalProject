from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
import string
from nltk.stem import SnowballStemmer
from glob import glob
import nltk
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

def tokenize(text, lang):
    stemmer = SnowballStemmer(lang)
    stop_words = set(stopwords.words(lang))

    tokens = word_tokenize(text)

    token_counter = Counter(tokens)

    tokens = [stemmer.stem(word) for word in tokens if \
            (word not in stop_words \
            and word not in string.punctuation)]
    return tokens

def word_2_vec(tokens, path, size=128, context=5):
    print('Inititalizing new model...')
    model = Word2Vec(sentences=[tokens], vector_size=size, window=context, min_count=1, workers=4)

    # write to path  
    model.save(path)
    return model.wv