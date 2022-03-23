from utils.data_loader import QuestionLoader
from utils.spacy_processing import tokenize
import json
from utils.similarity import get_max
source_file = '../../../data/preprocessing/preprocessing_v4.json'
labeled_file = '../../../data/abstract/seq2seq/sentence_labeled.json'


def lcs(X, Y, m=None, n=None):
    if m is None:
        m = len(X)
    if n is None:
        n = len(Y)
    if m == 0 or n == 0:
        return 0
    elif X[m - 1] == Y[n - 1]:
        return 1 + lcs(X, Y, m - 1, n - 1)
    else:
        return max(lcs(X, Y, m, n - 1), lcs(X, Y, m - 1, n))

def equal(v_1, v_2):
    l = lcs(v_1,v_2)
    return (l / len(v_1) + l / len(v_2)) / 2


def label(sen_tokens, sum_tokens):
    l = []
    for i in range(len(sum_tokens) - len(sen_tokens)):
        l.append(equal(v_1=sen_tokens, v_2=sum_tokens[i:i+len(sen_tokens)]))
    return get_max(l)

def labels(para_tokens, sum_tokens):
    rs = dict()
    for count in range(len(para_tokens)):
        rs[count] = label(para_tokens[count], sum_tokens)
    return rs



def create_multi_paragraph(questions):
    rs = dict()
    max_len_of_a_sen = 0
    spc = []
    max_len_of_a_para = 0

    for ques_id, ques in questions.items():
        q = dict()

        ext = ques.multi_ext_summ
        q['multi_ext_summ'] = ext
        q['question'] = ques.normalized_question

        sen_list = []
        token_list = []
        for ans_id, ans in ques.answers.items():
            for sen_id, sen in ans.sentences.items():
                s = sen.sentence
                t = sen.tokens
                sen_list.append(s)
                token_list.append(t)
                max_len_of_a_sen = max(max_len_of_a_sen, len(t))
                if len(t) > 512:
                    spc.append(s)

        q['label'] = labels(token_list, tokenize(ext))
        q['sentences'] = sen_list
        q['tokens'] = token_list
        rs[ques_id] = q
        max_len_of_a_para = max(max_len_of_a_para, len(sen_list))
    print('max length of a sentence: {} words'.format(max_len_of_a_sen))
    print('max length of a paragraph: {} sentences'.format(max_len_of_a_para))
    return rs


if __name__ == '__main__':
    questions = QuestionLoader().read_json(source_file).questions
    js = create_multi_paragraph(questions)
    json.dump(js, open(labeled_file, 'w+'), indent=4, sort_keys=True)
