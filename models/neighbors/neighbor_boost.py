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

    def get_score(self, obj, score_type):
        if score_type == 'final':
            return obj.final_score
        elif score_type == 'tfidf':
            return obj.tfidf_score
        elif score_type == 'lexrank':
            return obj.lexrank_score
        elif score_type == 'textrank':
            return obj.textrank_score
        elif score_type == 'query_base':
            return obj.query_base_score
        elif score_type == 'graph_base':
            return obj.graph_score
        elif score_type == 'ner':
            return obj.ner_score
        elif score_type == 'split_para':
            return obj.split_para_score
        elif score_type == 'random':
            return obj.random_score
        elif score_type == 'random_range':
            return obj.random_range_score
        return None

    def split_paragraphs(self, answer_id, ranks, relative_range=1, paragraph_threshold=0.5):
        paragraphs = dict()
        sentences = list()
        for rank in ranks:
            if rank.answer_id == answer_id:
                sentences.append((rank.sentence, rank.sentence_id))
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

    def train(self, total_ranks):
        self.scores = dict()
        for question_id, ranks in total_ranks.items():
            self.scores[question_id] = dict()
            for rank in ranks:
                if rank.answer_id not in self.scores[question_id]:
                    self.scores[question_id][rank.answer_id] = dict()
                self.scores[question_id][rank.answer_id][rank.sentence_id] = 0
        for question_id, all_ranks in total_ranks.items():
            ranks_dict = dict()
            for rank in all_ranks:
                if rank.answer_id not in ranks_dict:
                    ranks_dict[rank.answer_id] = list()
                ranks_dict[rank.answer_id].append(rank)
            if self.neighbor_type == 'relative':
                for answer_id, ranks in ranks_dict.items():
                    sorted_ranks = sorted(ranks, key=lambda obj: self.get_score(obj, self.score_type), reverse=True)
                    sorted_ranks = sorted_ranks[:min(self.limit_range, len(sorted_ranks))]
                    sorted_ranks = sorted(sorted_ranks, key=lambda obj: int(obj.sentence_id))
                    for i in range(1, len(sorted_ranks)):
                        left = int(sorted_ranks[i - 1].sentence_id)
                        right = int(sorted_ranks[i].sentence_id)
                        if right - left <= self.relative_range:
                            for j in range(left, right + 1):
                                self.scores[question_id][answer_id][str(j)] = self.get_score(sorted_ranks[i - 1],
                                                                                             self.score_type)
            elif self.neighbor_type == 'center':
                for answer_id, ranks in ranks_dict.items():
                    # sorted_ranks = sorted(ranks, key=lambda obj: self.get_score(obj, self.score_type), reverse=True)
                    # sorted_ranks = sorted_ranks[:min(self.limit_range, len(sorted_ranks))]
                    # sorted_ranks = sorted(sorted_ranks, key=lambda obj: int(obj.sentence_id))
                    # for rank in sorted_ranks:
                    #     sentence_id = int(rank.sentence_id)
                    #     for i in range(max(sentence_id - self.relative_range, 1),
                    #                    min(sentence_id + self.relative_range + 1, len(ranks) + 1)):
                    #         self.scores[question_id][answer_id][str(i)] = self.get_score(rank, self.score_type)
                    pairs = list()
                    for rank in ranks:
                        pairs.append((rank.sentence_id, self.get_score(rank, self.score_type)))
                    for pair in pairs:
                        sentence_id, score = int(pair[0]), pair[1]
                        for i in range(max(sentence_id - self.relative_range[0], 1),
                                       min(sentence_id + self.relative_range[1] + 1, len(ranks) + 1)):
                            self.scores[question_id][answer_id][str(i)] = max(
                                self.scores[question_id][answer_id][str(i)], score)
            elif self.neighbor_type == 'paragraph':
                for answer_id, ranks in ranks_dict.items():
                    paragraphs = self.split_paragraphs(answer_id=answer_id,
                                                       ranks=ranks,
                                                       relative_range=self.relative_range,
                                                       paragraph_threshold=self.threshold)
                    sorted_ranks = sorted(ranks, key=lambda obj: self.get_score(obj, self.score_type), reverse=True)
                    sorted_ranks = sorted_ranks[:min(self.limit_range, len(sorted_ranks))]
                    sorted_ranks = sorted(sorted_ranks, key=lambda obj: int(obj.sentence_id))
                    used_paragraph = dict()
                    for rank in sorted_ranks:
                        if paragraphs[rank.sentence_id] not in used_paragraph:
                            used_paragraph[paragraphs[rank.sentence_id]] = self.get_score(rank, self.score_type)
                    for rank in ranks:
                        if paragraphs[rank.sentence_id] in used_paragraph:
                            self.scores[question_id][answer_id][rank.sentence_id] = \
                                used_paragraph[paragraphs[rank.sentence_id]]
            else:
                for answer_id, ranks in ranks_dict.items():
                    for rank in ranks:
                        self.scores[question_id][answer_id][rank.sentence_id] = self.get_score(rank, self.score_type)
                        if self.boost_first is not None and int(rank.sentence_id) <= self.boost_first[0]:
                            self.scores[question_id][answer_id][rank.sentence_id] += self.boost_first[1]
                        elif self.boost_first is not None \
                                and len(ranks) - int(rank.sentence_id) + 1 <= self.boost_last[0]:
                            self.scores[question_id][answer_id][rank.sentence_id] += self.boost_last[1]
                self.scores = normalize_by_single_doc(self.scores)
        return self

    def predict_sentence(self, question_id, answer_id, sentence_id):
        return self.scores[question_id][answer_id][sentence_id]
