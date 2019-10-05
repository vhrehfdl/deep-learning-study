from konlpy.tag import Komoran
import json
import gensim


def make_corpus(input_file, output_file):
    txt_file = open(output_file, "w", encoding="utf-8")

    with open(input_file, 'r', encoding="utf-8") as f:
        text = f.readlines()
        num = 0

        for i in range(0, len(text)):
            sentence_list = text[i].strip()
            sentence = sentence_list.split('.')

            for j in range(0, len(sentence)):
                if len(sentence[j].strip()) > 30:

                    if num < 100000:
                        last_sentence = sentence[j].strip()
                        print(last_sentence)
                        txt_file.write(last_sentence + "\n")

                        num += 1

        print(num)


def make_token(input_file, output_file):
    komoran = Komoran()
    token_txt_file = open(output_file, "w", encoding="utf-8")
    list = []

    with open(input_file, 'r', encoding="utf-8") as f:
        text = f.readlines()
        num = 0

        for i in range(0, len(text)):
            sentence = text[i].strip()
            morphs = komoran.morphs(sentence)
            list.append(morphs)
            num += 1

        print(num)

        my_json_string = json.dumps(list, ensure_ascii=False)
        token_txt_file.write(my_json_string)


def main():
    make_corpus("Han.txt", "corpus.txt")
    make_token("corpus.txt", "corpus_token.txt")

    with open('corpus_token.txt', 'r', encoding="utf-8") as f:
        text = f.readlines()
        data = json.loads(text[0])

    embedding = gensim.models.Word2Vec(data, size=512, window=5, iter=5)
    embedding.save('han.word2vec.model')

    model = gensim.models.Word2Vec.load('han.word2vec.model')
    print(model.most_similar('문재인'))
    print(model.most_similar('한국'))
    print(model.most_similar('서울'))


if __name__ == '__main__':
    main()
