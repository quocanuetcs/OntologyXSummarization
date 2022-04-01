from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from entities import Answer, Question


def create_vocab(para, stop=stopwords.words('english')):
    iterable_docs = para
    if isinstance(para, Answer):
        iterable_docs = [sen.normalized_sentence for sen_id, sen in para.sentences.items()]
    else:
        if isinstance(para, Question):
            iterable_docs = list()
            iterable_docs.append(para.normalized_question)
            for ans_id, ans in para.answers.items():
                iterable_docs.extend([sen.normalized_sentence for sen_id, sen in ans.sentences.items()])
    victories = TfidfVectorizer(stop_words=stop)
    victories.fit_transform(iterable_docs)
    vocab = victories.vocabulary_
    rs = {}
    for key in vocab:
        # các từ càng xuất hiện nhiều thì điểm càng cao
        rs[key] = 1 / victories.idf_[vocab[key]]
    return rs


if __name__ == '__main__':
    print(create_vocab([
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]))
