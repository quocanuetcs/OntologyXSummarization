import spacy
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
# data = json.load(open('../../data/preprocessing/preprocessing_v2.json'))

class Sen:
    pass



class Data:
    pass


def create_vocab(iterable_docs):
    pass


if __name__ == '__main__':
    vectorizer = TfidfVectorizer()
    corpus = [
             'This is the first document.',
             'This document is the second document.',
             'And this is the third one.',
             'Is this the first document?',
        ]
    vocab = dict()
    X = vectorizer.fit_transform(corpus)
    Y = vectorizer.vocabulary_
    for c in X:
        print(c)