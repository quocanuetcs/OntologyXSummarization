from entities import SPACY
from models.base_model import BaseModel
from utils.data_loader import QuestionLoader
from math import sqrt

from utils.logger import get_logger
from utils.normalize import normalize_by_single_doc

logger = get_logger(__file__)


class TextRankScore(BaseModel):
    PHRASES_RATIO = 0.2

    def __init__(self, phrases_ratio=None):
        super().__init__()
        self.scores = dict()
        self.phrases_ratio = phrases_ratio if phrases_ratio is not None else self.PHRASES_RATIO
        SPACY.add_textrank_pipe()

    def train(self, questions):
        super().train(questions)
        for question_id, question in self.questions.items():
            logger.info('Training Textrank score of question {}'.format(question_id))
            self.scores[question_id] = dict()
            for answer_id, answer in question.answers.items():
                # Create corpus
                keys = dict()
                corpus = list()
                for sentence_id, sentence in answer.sentences.items():
                    keys[sentence_id] = len(corpus)
                    corpus.append(sentence.normalized_sentence)
                _text = ' '.join(corpus)
                doc = SPACY.spacy(_text)

                # create lists top-ranked phrases
                limit_phrases = round(self.phrases_ratio * len(doc._.phrases))
                phrase_id = 0
                unit_vector, phrase_text, chunks = list(), list(), list()
                for p in doc._.phrases:
                    unit_vector.append(p.rank)
                    phrase_text.append(p.text)
                    chunks.append(p.chunks)
                    phrase_id += 1
                    if phrase_id >= limit_phrases:
                        break

                # Normalize score
                sum_ranks = sum(unit_vector)
                unit_vector = [rank / sum_ranks for rank in unit_vector]

                sent_rank = dict()
                max_score, min_score = 0, 1000000000
                for sent_id in range(len(corpus)):
                    sum_sq = 0.0
                    for phrase_id in range(len(phrase_text)):
                        is_contained = False
                        for chunk in chunks[phrase_id]:
                            if chunk.lower_ in corpus[sent_id]:
                                is_contained = True
                                break
                        if not is_contained:
                            sum_sq += unit_vector[phrase_id] ** 2.0
                    sent_rank[sent_id] = 1 - sqrt(sum_sq)
                    max_score = max(max_score, sent_rank[sent_id])
                    min_score = min(min_score, sent_rank[sent_id])

                # Calculate Text Rank score
                self.scores[question_id][answer_id] = dict()
                for sentence_id, sentence in answer.sentences.items():
                    key = keys[sentence_id]
                    self.scores[question_id][answer_id][sentence_id] = sent_rank[key]
        self.scores = normalize_by_single_doc(self.scores)
        return self

    def predict_sentence(self, question_id, answer_id, sentence_id):
        return self.scores[question_id][answer_id][sentence_id]


if __name__ == '__main__':
    loader = QuestionLoader().read_json('../../data/preprocessing/small.json')
    textRank = TextRankScore()
    textRank.train({'1': loader.questions['1']})
    print(textRank.predict())