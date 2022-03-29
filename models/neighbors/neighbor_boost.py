from difflib import SequenceMatcher

from utils.logger import get_logger
from utils.normalize import normalize_by_single_doc

logger = get_logger(__file__)


def word_similarity(word_1, word_2):
    return SequenceMatcher(None, word_1, word_2).ratio()


def similarity(ner_1, ner_2, threshold=0.8):
    tokens_1 = ner_1.split(' ')
    tokens_2 = ner_2.split(' ')
    if len(tokens_1) + len(tokens_2) == 0:
        return 0
    valid_1, valid_2 = dict(), dict()
    for i in range(len(tokens_1)):
        for j in range(len(tokens_2)):
            if word_similarity(tokens_1[i], tokens_2[j]) >= threshold:
                valid_1[i], valid_2[j] = True, True
    return (len(valid_1) + len(valid_2)) / (len(tokens_1) + len(tokens_2))


# NER precision for sentence_2
def ner_ration(sentence_1, sentence_2, ner_threshold=0.4, word_threshold=0.8):
    if len(sentence_1.ners) + len(sentence_2.ners) == 0:
        return 0
    valid_1, valid_2 = dict(), dict()
    for i in range(len(sentence_1.ners)):
        for j in range(len(sentence_2.ners)):
            if similarity(ner_1=sentence_1.ners[i], ner_2=sentence_2.ners[j],
                          threshold=word_threshold) >= ner_threshold:
                valid_2[j] = True
    if len(sentence_2.ners) == 0:
        return 0
    return float(len(valid_2)) / len(sentence_2.ners)


SCORE_TYPES = ['tfidf', 'lexrank', 'textrank', 'query_base', 'graph_base', 'ner', 'split_para', 'final']


class NeighborBoost:
    SCORE_TYPE = 'tfidf'
    NEIGHBOR_TYPE = 'relative'
    LIMIT_RANGE = 10
    RELATIVE_RANGE = 20
    THRESHOLD = 0.5
    BOOST_FIRST = (0, 0)
    BOOST_LAST = (0, 0)

    def __init__(self,
                 score_type=None,
                 limit_range=None,
                 relative_range=None,
                 neighbor_type=None,
                 threshold=None,
                 boost_first=None,
                 boost_last=None):
        self.scores = None
        self.limit_range = limit_range if limit_range is not None else self.LIMIT_RANGE
        self.relative_range = relative_range if relative_range is not None else self.RELATIVE_RANGE
        self.score_type = score_type if score_type is not None else self.SCORE_TYPE
        self.neighbor_type = neighbor_type if neighbor_type is not None else self.NEIGHBOR_TYPE
        self.threshold = threshold if threshold is not None else self.THRESHOLD
        self.boost_first = boost_first if boost_first is not None else self.BOOST_FIRST
        self.boost_last = boost_last if boost_last is not None else self.BOOST_LAST
        self.input_scores = None

    def get_score(self, question_id, answer_id, sentence_id):
        sentence_score = self.input_scores[question_id][answer_id][sentence_id]
        if self.score_type == 'final':
            return sentence_score['final_score']
        elif self.score_type == 'tfidf':
            return sentence_score['tfidf_score']
        elif self.score_type == 'lexrank':
            return sentence_score['lexrank_score']
        elif self.score_type == 'query_base':
            return sentence_score['query_base_score']
        elif self.score_type == 'graph_base':
            return sentence_score['graph_score']
        elif self.score_type == 'ner':
            return sentence_score['ner_score']
        elif self.score_type == 'split_para':
            return sentence_score['split_para_score']
        return None

    def split_paragraphs(self, answer_id, sentence_list, relative_range=1, paragraph_threshold=0.5):
        paragraphs = dict()
        sentences = list()
        for sentence in sentence_list:
            sentences.append((sentence, sentence.id))

        paragraphs[sentences[0][1]] = 1
        for i in range(1, len(sentences)):
            last = i
            for j in range(max(i - relative_range, 0), i):
                if ner_ration(sentence_1=sentences[j][0], sentence_2=sentences[i][0],
                              ner_threshold=0.8, word_threshold=0.7) >= paragraph_threshold:
                    last = j
                    break
            if last == i:
                paragraphs[sentences[i][1]] = paragraphs[sentences[i - 1][1]] + 1
            else:
                for j in range(last + 1, i + 1):
                    paragraphs[sentences[j][1]] = paragraphs[sentences[last][1]]
        return paragraphs

    def train(self, questions, input_scores):
        self.input_scores = input_scores
        self.scores = dict()
        for question_id, question in questions.items():
            self.scores[question_id] = dict()
            for answer_id, answer in question.answers.items():
                if answer_id not in self.scores[question_id]:
                    self.scores[question_id][answer_id] = dict()
                for sentence_id, sentence in answer.sentences.items():
                    self.scores[question_id][answer_id][sentence_id] = 0

        for question_id, question in questions.items():
            answer_dicts = dict()
            for answer_id, answer in question.answers.items():
                if answer_id not in answer_dicts:
                    answer_dicts[answer_id] = list()
                for sentence_id, sentence in answer.sentences.items():
                    answer_dicts[answer_id].append(sentence)

            if self.neighbor_type == 'relative':
                for answer_id, sentence_list in answer_dicts.items():
                    sorted_sentences = sorted(sentence_list, key=lambda obj: self.get_score(question_id, answer_id, obj.id), reverse=True)
                    sorted_sentences = sorted_sentences[:min(self.limit_range, len(sorted_sentences))]
                    sorted_sentences = sorted(sorted_sentences, key=lambda obj: int(obj.id))
                    for i in range(1, len(sorted_sentences)):
                        left = int(sorted_sentences[i - 1].id)
                        right = int(sorted_sentences[i].id)
                        if right - left <= self.relative_range:
                            for j in range(left, right + 1):
                                self.scores[question_id][answer_id][str(j)] = self.get_score(question_id, answer_id, sorted_sentences[i - 1].id)

            elif self.neighbor_type == 'center':
                for answer_id, sentence_list in answer_dicts.items():
                    pairs = list()
                    for sentence in sentence_list:
                        pairs.append((sentence.id, self.get_score(question_id, answer_id, sentence.id)))
                    for pair in pairs:
                        sentence_id, score = int(pair[0]), pair[1]
                        for i in range(max(sentence_id - self.relative_range[0], 1),
                                       min(sentence_id + self.relative_range[1] + 1, len(sentence_list) + 1)):
                            self.scores[question_id][answer_id][str(i)] = max(
                                self.scores[question_id][answer_id][str(i)], score)

            elif self.neighbor_type == 'paragraph':
                for answer_id, sentence_list in answer_dicts.items():
                    paragraphs = self.split_paragraphs(answer_id=answer_id,
                                                       sentence_list=sentence_list,
                                                       relative_range=self.relative_range,
                                                       paragraph_threshold=self.threshold)
                    sorted_sentences = sorted(sentence_list, key=lambda obj: self.get_score(question_id, answer_id, obj.id), reverse=True)
                    sorted_sentences = sorted_sentences[:min(self.limit_range, len(sorted_sentences))]
                    sorted_sentences = sorted(sorted_sentences, key=lambda obj: int(obj.id))
                    used_paragraph = dict()
                    for sentence in sorted_sentences:
                        if paragraphs[sentence.id] not in used_paragraph:
                            used_paragraph[paragraphs[sentence.id]] = self.get_score(question_id, answer_id, sentence.id)
                    for sentence in sentence_list:
                        if paragraphs[sentence.id] in used_paragraph:
                            self.scores[question_id][answer_id][sentence.id] = \
                                used_paragraph[paragraphs[sentence.id]]
            else:
                for answer_id, sentence_list in answer_dicts.items():
                    for sentence in sentence_list:
                        self.scores[question_id][answer_id][sentence.id] = self.get_score(question_id, answer_id, sentence.id)
                        if self.boost_first is not None and int(sentence.id) <= self.boost_first[0]:
                            self.scores[question_id][answer_id][sentence.id] += self.boost_first[1]
                        elif self.boost_first is not None \
                                and len(sentence_list) - int(sentence.id) + 1 <= self.boost_last[0]:
                            self.scores[question_id][answer_id][sentence.id] += self.boost_last[1]
                self.scores = normalize_by_single_doc(self.scores)
        return self

    def predict_sentence(self, question_id, answer_id, sentence_id):
        return self.scores[question_id][answer_id][sentence_id]
