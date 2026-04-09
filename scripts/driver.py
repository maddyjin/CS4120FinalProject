from train_w2v import *
import argparse
from translate import *
import googletrans

def _parse_args():
    parser = argparse.ArgumentParser(description='driver.py')
    parser.add_argument('--lang', type=str, help='language of embeddings to be updated')
    parser.add_argument('--topic', type=str, help='name of topic to update embeddings with, or `all`')
    parser.add_argument('--fresh', type=bool, default=False, help='set True to overwrite embeddings and start fresh')
    parser.add_argument('--buffer', type=int, default=1000, help='number of characters to send to translator at once')
    args = parser.parse_args()
    return args

LANG_TO_CODE = googletrans.LANGCODES

if __name__ == '__main__':
    args = _parse_args()
    model_path = f"data/embeddings/{args.lang}_{args.topic}.model"


    text = read_file(args.topic, args.lang)

    full_translation = ''
    # use chunks of 1000
    while len(text) > 0:

        proposal = text[:args.buffer]
        if len(proposal) == args.buffer:
            last_space = proposal.rfind(' ')
            proposal = proposal[:last_space]

        trans = asyncio.run(translate_text(proposal, LANG_TO_CODE[args.lang]))
        full_translation += trans
        text = text[len(proposal):]
        print('Chars remaining:', len(text))

    tokens = tokenize(full_translation)
    vecs = word_2_vec(tokens, model_path, vs=128, context=5)

    print('First 10 keys:', vecs.index_to_key[:10])
    # print('Most similar to "lgbt":\n',vecs.most_similar('lgbt'))