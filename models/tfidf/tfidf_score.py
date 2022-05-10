from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from models.base_model import BaseModel
from utils.logger import get_logger
from utils.normalize import normalize_by_single_doc

logger = get_logger(__file__)

class TfIdfScore(BaseModel):
    def __init__(self,
                 keywords_boost=None,
                 keywords_type=None,
                 tfidf_ratio_threshold=None):
        super().__init__()
        self.word_scores = dict()
        self.scores = dict()
        self.keywords_boost = keywords_boost
        self.keywords_type = keywords_type
        self.tfidf_ratio_threshold = tfidf_ratio_threshold

    @staticmethod
    def custom_tokenize(text):
        return text

    def train(self, questions):
        super().train(questions)
        for question_id, question in self.questions.items():
            logger.info('Training Tf-Idf score of question {}'.format(question_id))
            # Create corpus
            keys = dict()
            corpus = list()
            mean_length = 0
            for answer_id, answer in question.answers.items():
                keys[answer_id] = dict()
                for sentence_id, sentence in answer.sentences.items():
                    keys[answer_id][sentence_id] = len(corpus)
                    tokens = sentence.tokens
                    corpus.append(tokens)
                    mean_length += len(tokens)
            mean_length /= len(corpus)

            # Create Tf-Idf vectorizer
            vectorizers = TfidfVectorizer(tokenizer=self.custom_tokenize, lowercase=False)
            vectorizers_matrix = vectorizers.fit_transform(corpus)
            feature_names = vectorizers.get_feature_names_out()
            feature_values = [0] * len(feature_names)
            for raw_vectorizer in vectorizers_matrix:
                vectorizer = raw_vectorizer.toarray().flatten()
                for i in range(len(feature_names)):
                    feature_values[i] = max(feature_values[i], vectorizer[i])

            # Calculate word's scores
            self.word_scores[question_id] = [{
                'word': feature_names[i],
                'score': feature_values[i]
            } for i in range(len(feature_names))]

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
                for sentence_id, sentence in answer.sentences.items():
                    key = keys[answer_id][sentence_id]
                    vectorizer = vectorizers_matrix[key].toarray().flatten()
                    length = 0
                    for i in range(len(vectorizer)):
                        if vectorizer[i] > 0:
                            length += 1
                            vectorizer[i] = feature_values[i]
                            if self.keywords_type == 'ner':
                                for ner in question.ners:
                                    if feature_names[i] in ner:
                                        vectorizer[i] += self.keywords_boost
                                        break
                            elif self.keywords_type == 'weight':
                                for keyword in list(question.keyword_weights):
                                    if (feature_names[i] in keyword) and (keyword in sentence.lemma.values()):
                                        vectorizer[i] = vectorizer[i] * question.keyword_weights[keyword]
                                        #vectorizer[i] = self.keywords_boost + vectorizer[i]*question.keyword_weights[keyword]
                                        #vectorizer[i] = self.keywords_boost + vectorizer[i] + question.keyword_weights[keyword]
                                        #vectorizer[i] = vectorizer[i] + question.keyword_weights[keyword]
                                        break
                    sorted_vectorizer = np.sort(np.array(vectorizer))[::-1]
                    sorted_vectorizer = sorted_vectorizer[sorted_vectorizer >= tfidf_threshold]
                    normalized_length = 1 + (max(length, mean_length) - mean_length) / mean_length
                    score = sorted_vectorizer.sum() / normalized_length
                    self.scores[question_id][answer_id][sentence_id] = score
        self.scores = normalize_by_single_doc(self.scores)
        return self

    def predict_sentence(self, question_id, answer_id, sentence_id):
        return self.scores[question_id][answer_id][sentence_id]


