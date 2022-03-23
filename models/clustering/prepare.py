from utils.similarity import *


def distance_in_para(sen_1, sen_2):
    return abs(sen_1.id - sen_2.id)


def get_overall_distance(sen_1, sen_2, vocab, fine_tune=[0.5, 0.5], ner_weight=2):
    rs = fine_tune[0] * word_weight_base(sen_1, sen_2, ner_weight=ner_weight, vocab=vocab)
    d = distance_in_para(sen_1, sen_2)
    if d != 0:
        rs += fine_tune[1] * 1 / d
    else:
        rs += 0.5
    return rs
