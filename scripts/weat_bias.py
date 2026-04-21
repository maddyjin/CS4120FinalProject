from wefe.word_embedding_model import WordEmbeddingModel
from wefe.query import Query
from wefe.metrics import WEAT
from gensim.models import Word2Vec

BASE_EMBEDDINGS_PATH = '../data/embeddings/'

def model_path(language, topic):
    return f'{BASE_EMBEDDINGS_PATH}{language}_{topic}.model'

english_model = Word2Vec.load(model_path('english', 'all'))
arabic_model = Word2Vec.load(model_path('arabic', 'all'))
#model = Word2Vec.load('../data/embeddings/english_philosophy.model')
wefe_model_english = WordEmbeddingModel(english_model.wv)
wefe_model_arabic = WordEmbeddingModel(arabic_model.wv)
#print(arabic_model.wv.index_to_key)

'''gender_query = Query(
    target_sets=[
        ['woman', 'female', 'feminine', 'girl', 'sister', 'she', 'her', 'hers', 'daughter'],
        ['man', 'male', 'masculine', 'boy', 'brother', 'he', 'him', 'his', 'son']
    ],
    attribute_sets=[
        ['irrational', 'emotion', 'emotional', 'intuitive', 'feeling'],
        ['rational', 'logic', 'logical', 'analytical', 'reasoning']
    ]
)'''
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
print(f'arabic weat score: {arabic_result['result']}, effect size: {arabic_result['effect_size']}, p-value: {arabic_result['p_value']}')