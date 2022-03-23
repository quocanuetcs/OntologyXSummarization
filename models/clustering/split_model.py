from utils.data_loader import QuestionLoader
from utils.normalize import normalize_by_single_doc
from entities import Answer
from models.querybase.prepare_data import create_vocab
from utils.statistic import *
from utils.logger import get_logger
from models.base_model import BaseModel

logger = get_logger(__file__)


def split_array(iterable, threshold):
    rs = []
    curr = -1
    for s, i in iterable:
        if curr == -1 or i < threshold:
            rs.append([s])
            curr += 1
        else:
            rs[curr].append(s)
    return rs


def split(para, sim, vocab=None, threshold=0.2, query=None):
    l = []
    sen = []
    if vocab is None:
        vocab = create_vocab(para)
    if isinstance(para, Answer):
        para = para.sentences
    for k, i in para.items():
        sen.append(i)
    l.append((sen[0], 0))
    for i in range(len(sen) - 1):
        l.append((sen[i + 1], sim(sen[i], sen[i + 1], vocab)))
    return split_array(l, threshold)


def get_score_of_para(list_para, vocab, query, sim):
    rs = dict()
    for para in list_para:
        e, d = standard_deviation([sim(sen, query, vocab) for sen in para])
        for sen in para:
            rs[sen.id] = e
    return rs


class SplitParaScore(BaseModel):
    THRESHOLD = 0.25

    def __init__(self, sim, threshold=None):
        super().__init__()
        self.data = dict()
        self.vocab = dict()
        self.sim = sim
        self.threshold = threshold if threshold is not None else self.THRESHOLD

    def train(self, questions, vocab_dict=None):
        super().train(questions)
        for ques_id, ques in self.questions.items():
            logger.info('Training Split_para score of question {}'.format(ques_id))
            query = ques
            self.vocab[ques_id] = create_vocab(ques) if vocab_dict is None else vocab_dict[ques_id]
            answer = dict()
            for ans_id, ans in ques.answers.items():
                answer[ans_id] = get_score_of_para(
                    split(para=ans, vocab=self.vocab[ques_id], threshold=self.threshold, query=query, sim=self.sim),
                    vocab=self.vocab[ques_id], query=query, sim=self.sim)
            # Normalize score
            max_score, min_score = 0, 1000000000
            for ans_id, ans in answer.items():
                for sen_id, sen in ans.items():
                    max_score = max(max_score, answer[ans_id][sen_id])
                    min_score = min(min_score, answer[ans_id][sen_id])
            for ans_id, ans in answer.items():
                for sen_id, sen in ans.items():
                    if min_score == max_score:
                        answer[ans_id][sen_id] = 1
                    else:
                        answer[ans_id][sen_id] = (answer[ans_id][sen_id] - min_score) / (max_score - min_score)
            self.data[ques_id] = answer
        self.data = normalize_by_single_doc(self.data)
        return self

    def predict_sentence(self, ques_id, answer_id, sentence_id):
        return self.data[ques_id][answer_id][sentence_id]


if __name__ == '__main__':
    loader = QuestionLoader().read_json('../../data/preprocessing/preprocessing_v3_section.json')
    lsamodel = SplitParaScore()
    lsamodel.train(loader.questions)
    print(lsamodel.predict())
