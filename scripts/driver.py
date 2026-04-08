from train_w2v import *
import argparse
from translate import *
import googletrans

def _parse_args():
    parser = argparse.ArgumentParser(description='driver.py')
    parser.add_argument('--lang', type=str, help='language of embeddings to be updated')
    parser.add_argument('--topic', type=str, help='name of topic to update embeddings with')
    parser.add_argument('--fresh', type=str, default=False, help='set True to overwrite embeddings and start fresh')
    args = parser.parse_args()
    return args

LANG_TO_CODE = googletrans.LANGCODES

if __name__ == '__main__':
    args = _parse_args()
    model_path = f"data/embeddings/{args.lang}.model"
    text = read_file(args.topic, args.lang)
    trans = asyncio.run(translate_text(text, LANG_TO_CODE[lang]))
    tokens = tokenize(text, args.lang)
    vecs = word_2_vec(tokens, model_path, args.fresh)

    print(vecs.index_to_key)