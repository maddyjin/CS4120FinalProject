from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize, sent_tokenize
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
        if topic == 'all':
            paths = glob(f'data/topics/*/{language}_*.txt', recursive=True)

            if len(paths) == 0:
                raise Exception(f'No files found for language {language}')

            txt = ''
            for path in paths:
                with open(path, 'rb') as f:
                    txt += f.read().decode().strip().lower()

        else:
            path = f"{topic}/{language}_{topic}"
            with open(f'data/topics/{path}.txt', 'rb') as f:
                txt = f.read().decode().strip().lower()

    except Exception as e:
        raise Exception(e)
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
    

def tokenize(text):
    lemma = wnl()
    stop_words = set(stopwords.words('english'))

    sentences = sent_tokenize(text)
    
    lemmas = []
    words = set()
    for sentence in sentences:

        tokens = word_tokenize(sentence)
        words = words.union(set(tokens))
        tokens = pos_tag(tokens, tagset='universal')
        
        sentence_lemmas = []
        for word, tag in tokens:
            if word not in stop_words and word not in string.punctuation:
                pos = get_wordnet_pos(tag)
                if pos:
                    sentence_lemmas.append(lemma.lemmatize(word, pos=pos).lower())
                else:
                    sentence_lemmas.append(lemma.lemmatize(word).lower())
        
        lemmas.append(sentence_lemmas)

    flat_lemmas = [item for sublist in lemmas for item in sublist]
    print(f'Reduced {len(words)} words to {len(set(flat_lemmas))} lemmas.')
    return lemmas

def word_2_vec(tokens, path, vs=128, context=5):
    # print('Inititalizing new model...')
    model = Word2Vec(sentences=tokens, vector_size=vs, window=context, min_count=1, workers=4)

    # write to path  
    model.save(path)
    return model.wv