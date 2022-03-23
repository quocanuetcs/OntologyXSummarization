from models.querybase.prepare_data import *
from models.base_model import BaseModel
from utils.data_loader import QuestionLoader
import matplotlib.pyplot as plt
from utils.logger import get_logger
from utils.normalize import normalize_by_single_doc

logger = get_logger(__file__)


class QueryBaseCore(BaseModel):
    NER_WEIGHT = 0

    def __init__(self, sim, ner_weight=None):
        super().__init__()
        self.para = None
        self.sim = sim
        self.query = None
        self.score = None
        self.embedding = None
        self.current_question_id = None
        self.current_data = None
        self.vocab = dict()
        self.data = dict()
        self.ner_weight = ner_weight if ner_weight is not None else self.NER_WEIGHT

    def train(self, questions, ques_id=None, answer_id=None, vocab_dict=None):
        super().train(questions)
        # self.embedding = BioBertEmbeddingLoader(questions)
        for ques_id, ques in self.questions.items():
            logger.info('Training Query-based score of question {}'.format(ques_id))
            self.vocab[ques_id] = create_vocab(para=ques) if vocab_dict is None else vocab_dict[ques_id]
            query = self.questions[ques_id]
            dict_ans = ques.answers  # load_answer(ques_id, ques)
            answer = dict()
            max_score, min_score = 0, 1000000000
            for ans_id, ans in dict_ans.items():
                answer[ans_id] = dict()
                for sen_id, sen in ans.sentences.items():
                    score = self.sim(sen, query, vocab=self.vocab[ques_id], ner_weight=self.NER_WEIGHT)
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
        # logger.info('Predict sentence {} of document {}'.format(sentence_id, answer_id))
        # a = self.questions[ques_id]
        # b = self.questions[ques_id].answers[answer_id].sentences[sentence_id]
        # if self.current_question_id is None or self.current_question_id != ques_id:
        #     self.current_question_id = ques_id
        #     self.current_data = self.embedding.get_question(ques_id)
        # data = self.current_data
        # vec_a = data['question_word_vector']
        # vec_b = data['answers'][answer_id][sentence_id]['word_vector']
        # return self.sim(sen_1=a, sen_2=b, v_1=vec_a, v_2=vec_b)
        return self.data[ques_id][answer_id][sentence_id]

    def info(self):
        plt.plot()
        score = []
        iden = []
        num = 0
        # print('question: '+ self.query.question)
        for k, v in self.score.items():
            num += 1
            score.append(v)
            iden.append(num)
            # print('score: ' + str(v))
            # print('sentence: '+ k)
        plt.xlabel('sentences')
        plt.ylabel(self.query.question)
        plt.plot(iden, score)
        plt.show()


if __name__ == '__main__':
    loader = QuestionLoader().read_json('../../data/preprocessing/preprocessing_v3.json')
    # questions = dict(itertools.islice(loader.questions.items(), 2))
    lsamodel = QueryBaseCore()
    lsamodel.train(loader.questions)
    # print(lsamodel.predict_sentence('1', '1_Answer1', '1'))
    # lsamodel.info()
