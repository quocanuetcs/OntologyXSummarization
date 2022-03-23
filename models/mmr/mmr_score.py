import nltk
import os
import math
import string
import re
import sentence
from nltk.corpus import stopwords
from utils.data_loader import QuestionLoader
from models.lexrank.lexrank_score import LexrankScore
from models.base_model import BaseModel


def TFs(sentences):
    # initialize tfs dictonary
    tfs = {}

    # for every sentence in document cluster
    for sent in sentences:
        # retrieve word frequencies from sentence object
        wordFreqs = sent.getWordFreq()

        # for every word
        for word in wordFreqs.keys():
            # if word already present in the dictonary
            if tfs.get(word, 0) != 0:
                tfs[word] = tfs[word] + wordFreqs[word]
            # else if word is being added for the first time
            else:
                tfs[word] = wordFreqs[word]
    return tfs


# ---------------------------------------------------------------------------------
# Description	: Function to find the inverse document frequencies of the words in
#				  the sentences present in the provided document cluster
# Parameters	: sentences, sentences of the document cluster
# Return 		: dictonary of word, inverse document frequency score
# ---------------------------------------------------------------------------------
def IDFs(sentences):
    N = len(sentences)
    idf = 0
    idfs = {}
    words = {}
    w2 = []

    # every sentence in our cluster
    for sent in sentences:

        # every word in a sentence
        for word in sent.getPreProWords():

            # not to calculate a word's IDF value more than once
            if sent.getWordFreq().get(word, 0) != 0:
                words[word] = words.get(word, 0) + 1

    # for each word in words
    for word in words:
        n = words[word]

        # avoid zero division errors
        try:
            w2.append(n)
            idf = math.log10(float(N) / n)
        except ZeroDivisionError:
            idf = 0

        # reset variables
        idfs[word] = idf

    return idfs


# ---------------------------------------------------------------------------------
# Description	: Function to find TF-IDF score of the words in the document cluster
# Parameters	: sentences, sentences of the document cluster
# Return 		: dictonary of word, TF-IDF score
# ---------------------------------------------------------------------------------
def TF_IDF(sentences):
    # Method variables
    tfs = TFs(sentences)
    idfs = IDFs(sentences)
    retval = {}

    # for every word
    for word in tfs:
        # calculate every word's tf-idf score
        tf_idfs = tfs[word] * idfs[word]

        # add word and its tf-idf score to dictionary
        if retval.get(tf_idfs, None) == None:
            retval[tf_idfs] = [word]
        else:
            retval[tf_idfs].append(word)

    return retval


# ---------------------------------------------------------------------------------
# Description	: Function to find the sentence similarity for a pair of sentences
#				  by calculating cosine similarity
# Parameters	: sentence1, first sentence
#				  sentence2, second sentence to which first sentence has to be compared
#				  IDF_w, dictinoary of IDF scores of words in the document cluster
# Return 		: cosine similarity score
# ---------------------------------------------------------------------------------
def sentenceSim(sentence1, sentence2, IDF_w):
    numerator = 0
    denominator = 0

    for word in sentence2.getPreProWords():
        numerator += sentence1.getWordFreq().get(word, 0) * sentence2.getWordFreq().get(word, 0) * IDF_w.get(word,
                                                                                                             0) ** 2

    for word in sentence1.getPreProWords():
        denominator += (sentence1.getWordFreq().get(word, 0) * IDF_w.get(word, 0)) ** 2

    # check for divide by zero cases and return back minimal similarity
    try:
        return numerator / math.sqrt(denominator)
    except ZeroDivisionError:
        return float("-inf")


# ---------------------------------------------------------------------------------
# Description	: Function to build a query of n words on the basis of TF-IDF value
# Parameters	: sentences, sentences of the document cluster
#				  IDF_w, IDF values of the words
#				  n, desired length of query (number of words in query)
# Return 		: query sentence consisting of best n words
# ---------------------------------------------------------------------------------
def buildQuery(sentences, TF_IDF_w, n):
    # sort in descending order of TF-IDF values
    scores = TF_IDF_w.keys()
    scores.sort(reverse=True)

    i = 0
    j = 0
    queryWords = []

    # select top n words
    while (i < n):
        words = TF_IDF_w[scores[j]]
        for word in words:
            queryWords.append(word)
            i = i + 1
            if (i > n):
                break
        j = j + 1

    # return the top selected words as a sentence
    return sentence.sentence("query", queryWords, queryWords)


# ---------------------------------------------------------------------------------
# Description	: Function to find the best sentence in reference to the query
# Parameters	: sentences, sentences of the document cluster
#				  query, reference query
#				  IDF, IDF value of words of the document cluster
# Return 		: best sentence among the sentences in the document cluster
# ---------------------------------------------------------------------------------
def bestSentence(sentences, query, IDF):
    best_sentence = None
    maxVal = float("-inf")

    for sent in sentences:
        similarity = sentenceSim(sent, query, IDF)

        if similarity > maxVal:
            best_sentence = sent
            maxVal = similarity
    sentences.remove(best_sentence)

    return best_sentence


# ---------------------------------------------------------------------------------
# Description	: Function to create the summary set of a desired number of words
# Parameters	: sentences, sentences of the document cluster
#				  best_sentnece, best sentence in the document cluster
#				  query, reference query for the document cluster
#				  summary_length, desired number of words for the summary
#				  labmta, lambda value of the MMR score calculation formula
#				  IDF, IDF value of words in the document cluster
# Return 		: name
# ---------------------------------------------------------------------------------
def makeScore(sentences, best_sentence, query, summary_length, lambta, IDF):
    summary = [best_sentence]
    sum_len = len(best_sentence.getPreProWords())

    MMRval = {}

    # keeping adding sentences until number of words exceeds summary length
    while (sum_len < summary_length):
        MMRval = {}

        for sent in sentences:
            MMRval[sent] = MMRScore(sent, query, summary, lambta, IDF)

        # maxxer = max(MMRval, key=MMRval.get)
        # summary.append(maxxer)
        # sentences.remove(maxxer)
        # sum_len += len(maxxer.getPreProWords())

    return MMRval


# ---------------------------------------------------------------------------------
# Description	: Function to calculate the MMR score given a sentence, the query
#				  and the current best set of sentences
# Parameters	: Si, particular sentence for which the MMR score has to be calculated
#				  query, query sentence for the particualr document cluster
#				  Sj, the best sentences that are already selected
#				  lambta, lambda value in the MMR formula
#				  IDF, IDF value for words in the cluster
# Return 		: name
# ---------------------------------------------------------------------------------
def MMRScore(Si, query, Sj, lambta, IDF):
    Sim1 = sentenceSim(Si, query, IDF)
    l_expr = lambta * Sim1
    value = [float("-inf")]

    for sent in Sj:
        Sim2 = sentenceSim(Si, sent, IDF)
        value.append(Sim2)

    r_expr = (1 - lambta) * max(value)
    MMR_SCORE = l_expr - r_expr

    return MMRScore


class mmrScore(BaseModel):
    def __init__(self):
        super().__init__()
        self.scores = dict()

    def train(self, questions):
        super().train(questions)
        lexRank = LexrankScore()
        lex_scores = lexRank.train(questions).scores
        for question_id, question in self.questions.items():
            corpus = list()
            scores = list()
            for answer_id, answer in question.answers.items():
                for sentence_id, sentence in answer.sentences.items():
                    corpus.append(sentence)
                    scores.append({
                        'answer_id': answer_id,
                        'sentence_id': sentence_id,
                        'score': lex_scores[question_id][answer_id][sentence_id]
                    })
            scores.sort(key=lambda obj: obj['score'], reverse=True)

            # for i in range(len(corpus)):
            #     for j in range(i, len(corpus)):
            #         if (scores[j]>scores[i]):
            #             scores[i],scores[j] = scores[j],scores[i]
            #             corpus[i], corpus[j] = corpus[j], corpus[i]

            # calculate TF, IDF and TF-IDF scores
            # TF_w 		= TFs(sentences)
            IDF_w = IDFs(corpus)
            TF_IDF_w = TF_IDF(corpus)

            # build query; set the number of words to include in our query
            query = buildQuery(corpus, TF_IDF_w, 10)

            # pick a sentence that best matches the query
            best1sentence = bestSentence(corpus, query, IDF_w)

            # build summary by adding more relevant sentences
            mmr_scores = makeScore(corpus, best1sentence, query, 200, 0.5, IDF_w)
        return self

    def predict_sentence(self, question_id, answer_id, sentence_id):
        return self.scores[question_id][answer_id][sentence_id]


if __name__ == '__main__':
    loader = QuestionLoader().read_json('../../data/small_.json')
    loader.normalize()
    mmrScores = mmrScore()
    mmr_scores = mmrScores.train(loader.questions)
