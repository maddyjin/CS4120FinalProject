from wefe.word_embedding_model import WordEmbeddingModel
from wefe.query import Query
from wefe.metrics import WEAT
from gensim.models import Word2Vec

BASE_EMBEDDINGS_PATH = '../data/embeddings/'

def model_path(language, topic):
    return f'{BASE_EMBEDDINGS_PATH}{language}_{topic}.model'

ENGLISH = 'english'
ARABIC = 'arabic'
GERMAN = 'german'
RUSSIAN = 'russian'
SPANISH = 'spanish'
LANGUAGES = [ENGLISH, ARABIC, GERMAN, RUSSIAN, SPANISH]

GENDER_QUERY_TARGETS = [
    ['woman', 'female', 'mother', 'daughter'],
    ['man', 'male', 'father', 'son']
]

language_models = {
    ENGLISH: Word2Vec.load(model_path(ENGLISH, 'all')),
    ARABIC: Word2Vec.load(model_path(ARABIC, 'all')),
    GERMAN: Word2Vec.load(model_path(GERMAN, 'all')),
    RUSSIAN: Word2Vec.load(model_path(RUSSIAN, 'all')),
    SPANISH: Word2Vec.load(model_path(SPANISH, 'all'))
}

wefe_models = {
    ENGLISH: WordEmbeddingModel(language_models[ENGLISH].wv),
    ARABIC: WordEmbeddingModel(language_models[ARABIC].wv),
    GERMAN: WordEmbeddingModel(language_models[GERMAN].wv),
    RUSSIAN: WordEmbeddingModel(language_models[RUSSIAN].wv),
    SPANISH: WordEmbeddingModel(language_models[SPANISH].wv),
}

#model = Word2Vec.load('../data/embeddings/english_philosophy.model')
#print(arabic_model.wv.index_to_key)

def run_query(language, targets, attribute_1, attribute_2):
    '''
    Returns the WEAT score, the size of the effect, and the p-value in a tuple.
    '''
    query = Query(
        target_sets=targets,
        attribute_sets=[
            attribute_1,
            attribute_2
        ]
    )
    metric = WEAT()
    result = metric.run_query(query, wefe_models[language], warn_not_found_words=False, lost_vocabulary_threshold=0.9999)
    #print(result)
    return result['result'], result['effect_size'], result['p_value'] if 'p_value' in result else None

def run_gender_query(language, attribute_1, attribute_2):
    '''
    Returns the WEAT score, the size of the effect, and the p-value in a tuple.
    '''
    return run_query(language, GENDER_QUERY_TARGETS, attribute_1, attribute_2)


RATIONALITY_ATTRIBUTE_SETS = [
    ['intuition', 'art', 'wisdom', 'feeling', 'believe', 'belief', 'faith', 'feel'],
    ['logic', 'reason', 'analysis', 'science', 'mathematics', 'intellectual', 'reasoning', 'think', 'philosophy', 'philosopher']
]


LEADERSHIP_ATTRIBUTE_SETS = [
    ['home', 'family', 'nurse', 'marriage', 'parent'],
    ['authority', 'leader', 'leadership', 'president', 'doctor', 'government']
]


for attribute_sets in [RATIONALITY_ATTRIBUTE_SETS, LEADERSHIP_ATTRIBUTE_SETS]:
    attributes_1, attributes_2 = attribute_sets
    print('===================')
    print(f'attribute sets\n- {attributes_1}\n- {attributes_2}')
    for language in LANGUAGES:
        weat, size, p_value = run_gender_query(language, attributes_1, attributes_2)
        print(language.upper())
        print(f'- {language} weat score: {weat}, effect size: {size}, p-value: {p_value}')