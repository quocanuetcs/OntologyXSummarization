from models.base_model import BaseModel
from utils.logger import get_logger
from utils.normalize import normalize_by_single_doc
logger = get_logger(__file__)


class QueryBaseCore(BaseModel):

    def __init__(self, sim):
        super().__init__()
        self.para = None
        self.sim = sim
        self.data = dict()

    def train(self, questions, ques_id=None, answer_id=None, vocab_dict=None):
        super().train(questions)
        for ques_id, ques in self.questions.items():
            logger.info('Training Query-based score of question {}'.format(ques_id))
            query = self.questions[ques_id]
            dict_ans = ques.answers  # load_answer(ques_id, ques)
            answer = dict()
            max_score, min_score = 0, 1000000000
            for ans_id, ans in dict_ans.items():
                answer[ans_id] = dict()
                for sen_id, sen in ans.sentences.items():
                    score = self.sim(sen, query)
                    answer[ans_id][sen_id] = score
                    max_score = max(max_score, score)
                    min_score = min(min_score, score)
            # Normalize score
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




