import random

from models.base_model import BaseModel
from utils.data_loader import QuestionLoader
from utils.logger import get_logger

logger = get_logger(__file__)


class RandomScore(BaseModel):
    NUM = 100

    def __init__(self, num=None):
        super().__init__()
        self.num = num if num is not None else self.NUM
        self.scores = dict()

    def train(self, questions):
        super().train(questions)
        for question_id, question in self.questions.items():
            logger.info('Training Random score of question {}'.format(question_id))
            self.scores[question_id] = dict()
            for answer_id, answer in question.answers.items():
                self.scores[question_id][answer_id] = dict()
                for sentence_id, sentence in answer.sentences.items():
                    self.scores[question_id][answer_id][sentence_id] = 0
                for i in range(self.num):
                    position = random.randint(1, len(answer.sentences))
                    score = random.random()
                    self.scores[question_id][answer_id][str(position)] += score

                # Normalize score
                max_score, min_score = 0, 1000000000
                for sentence_id, score in self.scores[question_id][answer_id].items():
                    max_score = max(max_score, score)
                    min_score = min(min_score, score)
                for sentence_id, sentence in answer.sentences.items():
                    score = self.scores[question_id][answer_id][sentence_id]
                    self.scores[question_id][answer_id][sentence_id] = (score - min_score) / (max_score - min_score)
        return self

    def predict_sentence(self, question_id, answer_id, sentence_id):
        return self.scores[question_id][answer_id][sentence_id]


if __name__ == '__main__':
    loader = QuestionLoader().read_json('../../data/preprocessing/preprocessing.json')
    tfidf = RandomScore()
    tfidf.train(loader.questions)
    print(tfidf.predict())
