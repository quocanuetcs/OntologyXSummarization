from lexrank import LexRank

from models.base_model import BaseModel
from models.lexrank.algorithms.summarizer import LexRank
from utils.data_loader import QuestionLoader
from utils.logger import get_logger
from utils.normalize import normalize_by_single_doc

logger = get_logger(__file__)

import itertools


class LexrankScore(BaseModel):
    THRESHOLD = None
    TF_FOR_ALL_QUESTION = False

    def __init__(self, threshold=None, tf_for_all_question=None):
        super().__init__()
        self.scores = dict()
        self.threshold = threshold if threshold is not None else self.THRESHOLD
        self.tf_for_all_question = tf_for_all_question if tf_for_all_question is not None else self.TF_FOR_ALL_QUESTION

    def train(self, questions):
        super().train(questions)
        if self.tf_for_all_question:
            documents = list()
            logger.info('Load document')
            for question_id, question in self.questions.items():
                for answer_id, answer in question.answers.items():
                    ans = list()
                    for sentence_id, sentence in answer.sentences.items():
                        ans.append(sentence)
                    documents.append(ans)
            lxr = LexRank(documents)

        for question_id, question in self.questions.items():
            local_docs = list()
            sentences = list()
            keys = dict()
            logger.info('Training Lexrank score of question {}'.format(question_id))
            for answer_id, answer in question.answers.items():
                ans = list()
                keys[answer_id] = dict()
                for sentence_id, sentence in answer.sentences.items():
                    keys[answer_id][sentence_id] = len(sentences)
                    ans.append(sentence)
                    sentences.append(sentence)

                if not self.tf_for_all_question: local_docs.append(ans)

            if not (self.tf_for_all_question): lxr = LexRank(local_docs)

            scores_cont = lxr.rank_sentences(
                sentences,
                fast_power_method=True,
                threshold=self.threshold
            )

            self.scores[question.id] = dict()
            for answer_id, answer in question.answers.items():
                self.scores[question.id][answer_id] = dict()
                for sentence_id, sentence in answer.sentences.items():
                    key = keys[answer_id][sentence_id]
                    self.scores[question.id][answer_id][sentence_id] = scores_cont[key]
        self.scores = normalize_by_single_doc(self.scores)
        return self

    def predict_sentence(self, question_id, answer_id, sentence_id):
        return self.scores[question_id][answer_id][sentence_id]


if __name__ == '__main__':
    loader = QuestionLoader().read_json('../../data/preprocessing/preprocessing_v3_section.json')
    questions = dict(itertools.islice(loader.questions.items(), 3))
    lexRank = LexrankScore(threshold=0.15, tf_for_all_question=True)
    lexRank.train(questions)
    print(lexRank.predict())
