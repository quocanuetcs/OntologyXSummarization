from models.base_model import BaseModel
from utils.normalize import normalize_by_single_doc
from utils.logger import get_logger
from models.querybase.prepare_data import create_vocab
logger = get_logger(__file__)


def create_score_dic(answer, vocab, sim, ner_weight=2):
    para = answer.sentences
    score = {}
    temp = {}
    _max = 0
    for i, s in para.items():
        score[i] = 0
        for j, r in para.items():
            if i != j:
                if (str(i) + '_' + str(j)) not in temp:
                    temp[str(i) + '_' + str(j)] = sim(para[i], para[j], vocab=vocab, ner_weight=ner_weight) / abs(
                        int(i) - int(j))
                    temp[str(j) + '_' + str(i)] = temp[str(i) + '_' + str(j)]
                score[i] += temp[str(i) + '_' + str(j)]
        if score[i] > _max:
            _max = score[i]

    for i, s in score.items():
        if _max == 0:
            _max = 1
        score[i] /= _max
    return score


class GraphScore(BaseModel):
    NER_WEIGHT = 0

    def __init__(self, sim, ner_weight=None):
        super().__init__()
        self.ner_weight = ner_weight if ner_weight is not None else self.NER_WEIGHT
        self.data = {}
        self.vocab = {}
        self.sim = sim

    def train(self, questions):
        super().train(questions)
        for ques_id, ques in self.questions.items():
            logger.info('Training Graph score of question {}'.format(ques_id))
            self.vocab[ques_id] = create_vocab(para=ques)
            dict_ans = ques.answers  # load_answer(ques_id, ques)
            answer = dict()
            for ans_id, ans in dict_ans.items():
                answer[ans_id] = create_score_dic(ans, vocab=self.vocab[ques_id], ner_weight=self.ner_weight,
                                                  sim=self.sim)
            self.data[ques_id] = answer
        self.data = normalize_by_single_doc(self.data)
        return self

    def predict_sentence(self, question_id, answer_id, sentence_id):
        result = self.data[question_id][answer_id][sentence_id]
        return result

