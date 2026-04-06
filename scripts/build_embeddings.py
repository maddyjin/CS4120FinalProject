from train_w2v import *


def main():
    langs = ['arabic', 'english', 'spanish', 'russian']
    topics = ['lgbt', 'philosophy', 'womensworldcup']


    for lang in langs:
        for topic in topics:
            model_path = f"data/embeddings/{lang}.model"
            text = read_file(topic, lang)
            tokens = tokenize(text, lang)
            vecs = word_2_vec(tokens, model_path)


if __name__ == '__main__':
    main()