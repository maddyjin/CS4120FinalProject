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

GENDER_QUERY_TARGETS = {
    ENGLISH: [
        ['woman', 'female',],
        ['man', 'male',]
    ],
    ARABIC: [
        ['female'],
        ['male']
    ],
    GERMAN: [
        ['woman', 'female',],
        ['man', 'male',]
    ],
    RUSSIAN: [
        ['woman'],
        ['man']
    ],
    SPANISH: [
        ['woman', 'female',],
        ['man', 'male',]
    ],
}

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
    result = metric.run_query(query, wefe_models[language], warn_not_found_words=True)
    #print(result)
    return result['result'], result['effect_size'], result['p_value'] if 'p_value' in result else None

def run_gender_query(language, attribute_1, attribute_2):
    '''
    Returns the WEAT score, the size of the effect, and the p-value in a tuple.
    '''
    return run_query(language, GENDER_QUERY_TARGETS[language], attribute_1, attribute_2)


RATIONALITY_ATTRIBUTE_SETS = {
    ENGLISH: [
        ['intuition', 'art', 'wisdom', 'feeling', 'believe', 'belief', 'faith', 'feel'],
        ['logic', 'reason', 'analysis', 'science', 'mathematics', 'intellectual', 'reasoning', 'think']
    ],
    ARABIC: [
        ['intuition', 'art', 'wisdom', 'feeling', 'believe', 'belief', 'faith', 'feel'],
        ['logic', 'reason', 'analysis', 'science', 'mathematics', 'intellectual', 'reasoning', 'think']
    ],
    GERMAN: [
        ['art', 'wisdom', 'believe', 'belief'],
        ['logic', 'reason', 'analysis', 'science', 'mathematics', 'intellectual', 'reasoning', 'think']
    ],
    RUSSIAN: [
        ['art', 'wisdom', 'feeling', 'believe', 'belief', 'feel'],
        ['logic', 'reason', 'analysis', 'science', 'mathematics', 'intellectual', 'reasoning', 'think']
    ],
    SPANISH: [
        ['intuition', 'art', 'wisdom', 'feeling', 'believe', 'belief', 'faith', 'feel'],
        ['logic', 'reason', 'analysis', 'science', 'mathematics', 'intellectual', 'reasoning', 'think']
    ],
}

for language in LANGUAGES:
    attributes_1, attributes_2 = RATIONALITY_ATTRIBUTE_SETS[language]
    weat, size, p_value = run_gender_query(language, attributes_1, attributes_2)
    print(language.upper())
    print(f'- attribute sets\n   - {attributes_1}\n   - {attributes_2}')
    print(f'- {language} weat score: {weat}, effect size: {size}, p-value: {p_value}')


'''gender_query = Query(
    target_sets=[
        ['woman', 'female', 'feminine', 'girl', 'sister', 'she', 'her', 'hers', 'daughter'],
        ['man', 'male', 'masculine', 'boy', 'brother', 'he', 'him', 'his', 'son']
    ],
    attribute_sets=[
        ['irrational', 'emotion', 'emotional', 'intuitive', 'feeling'],
        ['rational', 'logic', 'logical', 'analytical', 'reasoning']
    ]
)
gender_query_english = Query(
    target_sets=[
        ['woman', 'female',],
        ['man', 'male',]
    ],
    attribute_sets=[
        ['intuition', 'art', 'wisdom', 'feeling', 'believe', 'belief', 'faith'],
        ['logic', 'reason', 'analysis', 'science', 'mathematics', 'intellectual', 'reasoning']
    ]
)
gender_query_arabic = Query(
    target_sets=[
        ['female'],
        ['male']
    ],
    attribute_sets=[
        ['intuition', 'art', 'wisdom', 'feeling', 'believe', 'belief', 'faith'],
        ['logic', 'reason', 'analysis', 'science', 'mathematics', 'intellectual', 'reasoning']
    ]
)

#print(gender_query_english)

metric = WEAT()
english_result = metric.run_query(gender_query_english, wefe_model_english)
print(f'english query target sets: {gender_query_english.target_sets}')
print(f'english query attribute sets: {gender_query_english.attribute_sets}')
print(f'english weat score: {english_result['result']}, effect size: {english_result['effect_size']}, p-value: {english_result['p_value']}')
print()

arabic_result = metric.run_query(gender_query_arabic, wefe_model_arabic)
print(f'arabic query target sets: {gender_query_arabic.target_sets} (the word man did not show up in the embedding)')
print(f'arabic query attribute sets: {gender_query_arabic.attribute_sets}')
print(f'arabic weat score: {arabic_result['result']}, effect size: {arabic_result['effect_size']}, p-value: {arabic_result['p_value']}')'''