import math
from collections import Counter, defaultdict
import numpy as np
from models.lexrank.algorithms.power_method import stationary_distribution
from sklearn.feature_extraction.text import TfidfVectorizer

def custom_tokenize(text):
    return text

class LexRank:
    def __init__(
        self,
        documents,
        include_new_words=True,
    ):
        self.include_new_words = include_new_words
        self.idf_score = self._calculate_idf(documents)

    def _calculate_idf(self, documents):
        corpus = list()
        for document in documents:
            for sentence in document:
                tokens = sentence.tokens
                corpus.append(tokens)

        # Create Tf-Idf vectorizer
        vectorizers = TfidfVectorizer(tokenizer=custom_tokenize, lowercase=False)
        vectorizers.fit_transform(corpus)
        vocab = vectorizers.vocabulary_
        rs = {}
        for key in vocab:
            rs[key] = 1 / vectorizers.idf_[vocab[key]]
        return rs


    def rank_sentences(
        self,
        question,
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
            #count = Counter(self.tokenize_sentence(sentence))
            count = Counter(sentence.tokens)
            tf_scores.append(count)

        similarity_matrix = self._calculate_similarity_matrix(tf_scores,question)

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


    def _calculate_similarity_matrix(self, tf_scores, question):
        length = len(tf_scores)

        similarity_matrix = np.zeros([length] * 2)

        for i in range(length):
            for j in range(i, length):
                similarity = self._idf_modified_cosine(tf_scores, i, j, question)

                if similarity:
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity

        return similarity_matrix

    def _idf_modified_cosine(self, tf_scores, i, j, question):
        if i == j:
            return 1

        tf_i, tf_j = tf_scores[i], tf_scores[j]
        words_i, words_j = set(tf_i.keys()), set(tf_j.keys())

        nominator = 0

        for word in words_i & words_j:
            idf = self.idf_score[word]
            nominator += question.keyword_weights.get(word,1)*tf_i[word] * tf_j[word] * idf ** 2

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
