import numpy as np
import torch
import re
from collections import Counter
from nltk.corpus import stopwords
from rouge import Rouge
from torch.nn import CosineSimilarity
from utils.embedding_process.get_embedding_data import Dictionary, FOLDER_HOLDING_EMBEDDING_DATA

STOPWORDS = stopwords.words('english')
BERT_DICTIONARY = Dictionary(MAX_THREADS=4, path_to_folder=FOLDER_HOLDING_EMBEDDING_DATA)
COS = CosineSimilarity()


def get_max(iterable):
    rs = None
    for item in iterable:
        if rs is None or rs < item:
            rs = item
    return rs


def average(iterable):
    rs = 0
    for item in iterable:
        rs += item / len(iterable)
    return rs


def cosine(vec_1, vec_2):
    try:
        rs = COS(torch.tensor([vec_1]), torch.tensor([vec_2])).tolist()[0]
        return rs
    except:
        return -1


def euclid_distance(vec_1, vec_2):
    return np.linalg.norm(np.array(vec_1) - np.array(vec_2))


def rouge_sim(sen_1, sen_2):
    return Rouge().get_scores(sen_1.sentence, sen_2.sentence)


def bert_base(sen_1, sen_2, vocab=None, opt="MAX", ner_weight=0):
    if not isinstance(sen_1, str):
        try:
            sen_1 = sen_1.sentence
        except Exception:
            sen_1 = sen_1.question
    if not isinstance(sen_2, str):
        try:
            sen_2 = sen_2.sentence
        except Exception:
            sen_2 = sen_2.question
    vec_1 = BERT_DICTIONARY.get_embedding_of(sen_1)
    vec_2 = BERT_DICTIONARY.get_embedding_of(sen_2)
    return cosine(vec_1, vec_2)


def word_weight_base(sen_1, sen_2, vocab=None, opt="MAX", ner_weight=0):
    # vocab is a dictionary (such as tf-idf) format like:
    # {
    #     'word_1': weight_1
    #     ....
    # }
    # or a list of tuple:
    # [
    #     (word,weight)
    #     ....
    # ]
    ner_1 = ner_2 = None
    if isinstance(sen_1, str):
        vec_1 = re.findall(r'\w+', sen_1.lower())
    else:
        vec_1 = sen_1.tokens
        ner_1 = Counter(sen_1.ners)
    if isinstance(sen_2, str):
        vec_2 = re.findall(r'\w+', sen_2.lower())
    else:
        vec_2 = sen_2.tokens
        ner_2 = Counter(sen_2.ners)
    vec_1 = Counter(vec_1)
    vec_2 = Counter(vec_2)
    sc_1 = n_1 = 0
    sc_2 = n_2 = 0
    if isinstance(vocab, list):
        temp = vocab
        vocab = dict()
        for word, weight in temp:
            vocab[word] = weight
    if ner_1 is not None and ner_2 is not None:
        for ner in ner_1:
            if ner in ner_2:
                sc_1 += ner_1[ner] * ner_weight
                sc_2 += ner_2[ner] * ner_weight
        for ner in ner_1:
            n_1 += ner_1[ner] * ner_weight
        for ner in ner_2:
            n_2 += ner_2[ner] * ner_weight
    for word_1, time_1 in vec_1.items():
        if word_1 in vocab:
            if word_1 in vec_2:
                sc_1 += time_1 * vocab[word_1]
            n_1 += time_1 * vocab[word_1]
    for word_2, time_2 in vec_2.items():
        if word_2 in vocab:
            if word_2 in vec_1:
                sc_2 += time_2 * vocab[word_2]
            n_2 += time_2 * vocab[word_2]
    if n_1 * n_2 == 0:
        return 0
    if opt.lower() == 'max':
        return get_max([sc_1 / n_1, sc_2 / n_2])
    return average([sc_1 / n_1, sc_2 / n_2])
