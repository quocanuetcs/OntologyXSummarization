from math import log

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

from models.base_model import BaseModel
from utils import preprocessing
from utils.data_loader import QuestionLoader
from utils.logger import get_logger

logger = get_logger(__file__)


class TfIdfScoreV2(BaseModel):
    KEYWORDS_BOOST = 1
    KEYWORDS_TYPE = 'ner'
    TFIDF_RATIO_THRESHOLD = 0.2

    def __init__(self,
                 keywords_boost=None,
                 keywords_type=None,
                 tfidf_ratio_threshold=None):
        super().__init__()
        self.word_scores = dict()
        self.scores = dict()
        self.keywords_boost = keywords_boost if keywords_boost is not None else self.KEYWORDS_BOOST
        self.keywords_type = keywords_type if keywords_type is not None else self.KEYWORDS_TYPE
        self.tfidf_ratio_threshold = tfidf_ratio_threshold if tfidf_ratio_threshold is not None \
            else self.TFIDF_RATIO_THRESHOLD
        self.idf_cache = dict()

    @staticmethod
    def custom_tokenize(text):
        return text

    def idf(self, word, documents):
        if word in self.idf_cache:
            return self.idf_cache[word]
        cnt = 0
        for tokens in documents:
            if word in tokens:
                cnt += 1
        self.idf_cache[word] = log(float(len(documents)) + 1) / (cnt + 1)
        return self.idf_cache[word]

    def train(self, questions):
        super().train(questions)
        # Prepare full size of corpus
        documents = list()
        for question_id, question in self.questions.items():
            for answer_id, answer in question.answers.items():
                for sentence_id, sentence in answer.sentences.items():
                    tokens = sentence.bert_tokens
                    tokens = preprocessing.remove_stopwords(tokens)
                    tokens = preprocessing.remove_numbers(tokens)
                    # tokens = preprocessing.stemming(tokens)
                    documents.append(tokens)
        for question_id, question in self.questions.items():
            logger.info('Training Tf-Idf V2 score of question {}'.format(question_id))
            # Create corpus
            keys = dict()
            corpus = list()
            mean_length = 0
            for answer_id, answer in question.answers.items():
                keys[answer_id] = dict()
                for sentence_id, sentence in answer.sentences.items():
                    keys[answer_id][sentence_id] = len(corpus)
                    tokens = sentence.bert_tokens
                    tokens = preprocessing.remove_stopwords(tokens)
                    tokens = preprocessing.remove_numbers(tokens)
                    # tokens = preprocessing.stemming(tokens)
                    corpus.append(tokens)
                    mean_length += len(tokens)
            mean_length /= len(corpus)

            # Create Tf-Idf vectorizer
            vectorizers = CountVectorizer(tokenizer=self.custom_tokenize, lowercase=False)
            vectorizers_matrix = vectorizers.fit_transform(corpus)
            feature_names = vectorizers.get_feature_names()
            feature_values = [0.0] * len(feature_names)
            key = 0
            for raw_vectorizer in vectorizers_matrix:
                vectorizer = raw_vectorizer.toarray().flatten()
                for i in range(len(feature_names)):
                    if len(corpus[key]) == 0:
                        feature_values[i] = 0
                    else:
                        feature_values[i] = max(feature_values[i], float(vectorizer[i]) / len(corpus[key]))
                key += 1

            # Calculate word's scores
            self.word_scores[question_id] = [{
                'word': feature_names[i],
                'score': feature_values[i]
            } for i in range(len(feature_names))]

            # Boost question pos tags for nouns and verbs in question
            for i in range(len(self.word_scores[question_id])):
                word_score = self.word_scores[question_id][i]
                if word_score['word'] in question.nouns or word_score['word'] in question.verbs:
                    self.word_scores[question_id][i]['score'] += self.keywords_boost
                    feature_values[i] += self.keywords_boost
                    break

            # Sort list of scores
            self.word_scores[question_id] = sorted(self.word_scores[question_id],
                                                   key=lambda obj: obj['score'],
                                                   reverse=True)

            # Calculate threshold for tfidf
            threshold_position = int(len(self.word_scores[question_id]) * self.tfidf_ratio_threshold)
            tfidf_threshold = self.word_scores[question_id][threshold_position]['score']

            # Calculate Tf-Idf score from vectorizer
            self.scores[question_id] = dict()
            for answer_id, answer in question.answers.items():
                self.scores[question.id][answer_id] = dict()
                max_score, min_score = 0, 1000000000
                for sentence_id, sentence in answer.sentences.items():
                    key = keys[answer_id][sentence_id]
                    # Calculate vectorizer
                    vectorizer = vectorizers_matrix[key].toarray().flatten()
                    length = 0
                    for i in range(len(vectorizer)):
                        if vectorizer[i] > 0:
                            length += 1
                            # tf x idf of all documents
                            vectorizer[i] = feature_values[i] * self.idf(feature_names[i], documents)
                            if self.keywords_type == 'pos-tag' \
                                    and (feature_names[i] in question.nouns or feature_names[i] in question.verbs):
                                vectorizer[i] += self.keywords_boost
                            elif self.keywords_type == 'ner':
                                for ner in question.ners:
                                    if feature_names[i] in ner:
                                        vectorizer[i] += self.keywords_boost
                                        break
                    sorted_vectorizer = np.sort(np.array(vectorizer))[::-1]
                    sorted_vectorizer = sorted_vectorizer[sorted_vectorizer >= tfidf_threshold]
                    normalized_length = 1 + (max(length, mean_length) - mean_length) / mean_length
                    score = sorted_vectorizer.sum() / normalized_length
                    self.scores[question_id][answer_id][sentence_id] = score
                    max_score = max(max_score, score)
                    min_score = min(min_score, score)

                # Normalize score
                for sentence_id, sentence in answer.sentences.items():
                    score = self.scores[question_id][answer_id][sentence_id]
                    self.scores[question_id][answer_id][sentence_id] = (score - min_score) / (max_score - min_score)
        return self

    def predict_sentence(self, question_id, answer_id, sentence_id):
        return self.scores[question_id][answer_id][sentence_id]


if __name__ == '__main__':
    loader = QuestionLoader().read_json('../../data/preprocessing/preprocessing.json')
    import itertools

    questions = dict(itertools.islice(loader.questions.items(), 30))
    tfidf = TfIdfScoreV2()
    tfidf.train(questions)
    print(tfidf.predict())
