from difflib import SequenceMatcher

from utils.data_loader import *
from utils.data_loader import QuestionLoader
from models.base_model import BaseModel


# def ner_ration(sen, vocab=None, rate_f=None):
#     if vocab is None:
#         return len(sen.ners)
#     rs = 0
#     for ner in sen.ners:
#         rs += rate_f(ner, vocab)
#     return rs
from utils.normalize import normalize_by_single_doc


def word_similarity(word_1, word_2):
    return SequenceMatcher(None, word_1, word_2).ratio()


def similarity(ner_1, ner_2, threshold=0.8):
    tokens_1 = ner_1.split(' ')
    tokens_2 = ner_2.split(' ')
    if len(tokens_1) + len(tokens_2) == 0:
        return 0
    valid_1, valid_2 = dict(), dict()
    for i in range(len(tokens_1)):
        for j in range(len(tokens_2)):
            if word_similarity(tokens_1[i], tokens_2[j]) >= threshold:
                valid_1[i], valid_2[j] = True, True
    return (len(valid_1) + len(valid_2)) / (len(tokens_1) + len(tokens_2))


def ner_ration(sentence, question, ner_threshold=0.4, word_threshold=0.8):
    if len(sentence.ners) + len(question.ners) == 0:
        return 0
    valid_1, valid_2 = dict(), dict()
    for i in range(len(sentence.ners)):
        for j in range(len(question.ners)):
            if similarity(ner_1=sentence.ners[i], ner_2=question.ners[j], threshold=word_threshold) >= ner_threshold:
                valid_1[i], valid_2[j] = True, True
    return float(len(valid_1) + len(valid_2)) / (len(sentence.ners) + len(question.ners))


class NerRatio(BaseModel):
    NER_THRESHOLD = 0.6
    WORD_THRESHOLD = 0.6

    def __init__(self, ner_threshold=None, word_threshold=None):
        super().__init__()
        self.data = dict()
        self.ner_threshold = ner_threshold if ner_threshold is not None else self.NER_THRESHOLD
        self.word_threshold = word_threshold if word_threshold is not None else self.WORD_THRESHOLD

    def train(self, questions):
        super().train(questions)
        for ques_id, ques in self.questions.items():
            logger.info('Training NER score of question {}'.format(ques_id))
            q = dict()
            for ans_id, ans in ques.answers.items():
                m = 0
                a = dict()
                for sen_id, sen in ans.sentences.items():
                    a[sen_id] = ner_ration(sen, ques,
                                           ner_threshold=self.ner_threshold,
                                           word_threshold=self.word_threshold)
                    m = max(m, a[sen_id])
                if m != 0:
                    for key in a:
                        a[key] /= m
                q[ans_id] = a
            self.data[ques_id] = q
        self.data = normalize_by_single_doc(self.data)
        return self

    def predict_sentence(self, question_id, answer_id, sentence_id):
        return self.data[question_id][answer_id][sentence_id]


if __name__ == '__main__':
    ques = QuestionLoader().read_json('../../data/preprocessing/preprocessing.json').questions
    model = NerRatio().train({'1': ques['1']})
    print(model.predict())
