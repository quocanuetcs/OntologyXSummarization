import math
from collections import Counter, defaultdict

import numpy as np

from models.lexrank.algorithms.power_method import stationary_distribution
from models.lexrank.utils.text import tokenize
from lexrank.utils.text import tokenize as auto_tokenize
from utils.preprocessing import tokenize as tkn
from nltk.corpus import stopwords

class LexRank:
    def __init__(
        self,
        documents,
        include_new_words=True,
    ):
        self.include_new_words = include_new_words
        self.idf_score = self._calculate_idf(documents)

    def _calculate_idf(self, documents):
        bags_of_words = []

        for doc in documents:
            doc_words = set()

            for sentence in doc:
                #words = self.tokenize_sentence(sentence)
                words = tkn(sentence.normalized_sentence)
                doc_words.update(words)

            if doc_words:
                bags_of_words.append(doc_words)

        if not bags_of_words:
            raise ValueError('documents are not informative')

        doc_number_total = len(bags_of_words)

        if self.include_new_words:
            default_value = math.log(doc_number_total + 1)

        else:
            default_value = 0

        idf_score = defaultdict(lambda: default_value)

        for word in set.union(*bags_of_words):
            doc_number_word = sum(1 for bag in bags_of_words if word in bag)
            idf_score[word] = math.log(doc_number_total / doc_number_word)

        return idf_score

    def rank_sentences(
        self,
        sentences,
        threshold=.03,
        fast_power_method=True,
    ):
        if not (
            threshold is None or
            isinstance(threshold, float) and 0 <= threshold < 1
        ):
            raise ValueError(
                '\'threshold\' should be a floating-point number '
                'from the interval [0, 1) or None',
            )

        tf_scores = []

        for sentence in sentences:
            count = Counter(self.tokenize_sentence(sentence))
            tf_scores.append(count)

        similarity_matrix = self._calculate_similarity_matrix(tf_scores)

        if threshold is None:
            markov_matrix = self._markov_matrix(similarity_matrix)

        else:
            markov_matrix = self._markov_matrix_discrete(
                similarity_matrix,
                threshold=threshold,
            )

        scores = stationary_distribution(
            markov_matrix,
            increase_power=fast_power_method,
            normalized=False,
        )

        return scores

    # def tokenize_sentence(self, sentence):
    #     tokens = tokenize(sentence)
    #     return tokens


    def tokenize_sentence(self, sentence):
        stw = stopwords.words('english')
        tokens = auto_tokenize(
            sentence.normalized_sentence,
            stw,
            False,
            False,
            False,
        )
        return tokens

    def _calculate_similarity_matrix(self, tf_scores):
        length = len(tf_scores)

        similarity_matrix = np.zeros([length] * 2)

        for i in range(length):
            for j in range(i, length):
                similarity = self._idf_modified_cosine(tf_scores, i, j)

                if similarity:
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity

        return similarity_matrix

    def _idf_modified_cosine(self, tf_scores, i, j):
        if i == j:
            return 1

        tf_i, tf_j = tf_scores[i], tf_scores[j]
        words_i, words_j = set(tf_i.keys()), set(tf_j.keys())

        nominator = 0

        for word in words_i & words_j:
            idf = self.idf_score[word]
            nominator += tf_i[word] * tf_j[word] * idf ** 2

        if math.isclose(nominator, 0):
            return 0

        denominator_i, denominator_j = 0, 0

        for word in words_i:
            tfidf = tf_i[word] * self.idf_score[word]
            denominator_i += tfidf ** 2

        for word in words_j:
            tfidf = tf_j[word] * self.idf_score[word]
            denominator_j += tfidf ** 2

        similarity = nominator / math.sqrt(denominator_i * denominator_j)

        return similarity

    def _markov_matrix(self, similarity_matrix):
        row_sum = similarity_matrix.sum(axis=1, keepdims=True)

        return similarity_matrix / row_sum

    def _markov_matrix_discrete(self, similarity_matrix, threshold):
        markov_matrix = np.zeros(similarity_matrix.shape)

        for i in range(len(similarity_matrix)):
            columns = np.where(similarity_matrix[i] > threshold)[0]
            markov_matrix[i, columns] = 1 / len(columns)

        return markov_matrix
