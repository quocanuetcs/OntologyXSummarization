
class PREPROCESSING_CONFIG:
    def __init__(self):
        self.name = 'validation'


class TFIDF_WEIGHT:
    def __init__(self):
        self.ratio_threshold = 0.9
        self.keywords_boost = 3.0
        self.keywords_type = 'ner'

class LEXRANK_WEIGHT:
    def __init__(self):
        self.threshold = 0.065
        self.tf_for_all_question = True

class QUERY_BASE_WEIGHT:
    def __init__(self):
        self.query_ner_weight = 0

class NER_WEIGHT:
    def __init__(self):
        self.ner_threshold = 0.7
        self.word_threshold = 0.8

class SPLIT_WEIGHT:
    def __init__(self):
        self.split_threshold = 0.1

class GRAPH_WEIGHT:
    def __init__(self):
        self.ner_weight = 0

class FINAL_WEIGHT:
    def __init__(self):
        self.tfidf = 7
        self.lexrank = 3
        self.graph = 3
        self.query_base = 5
        self.ner = 5
        self.split = 10
        self.total_weight = self.tfidf + self.lexrank + self.graph + self.query_base + self.ner + self.split

class SENTENCE_SCORING:
    def __init__(self):
        self.tfidf = TFIDF_WEIGHT()
        self.lexrank = LEXRANK_WEIGHT()
        self.query_base = QUERY_BASE_WEIGHT()
        self.ner = NER_WEIGHT()
        self.split = SPLIT_WEIGHT()
        self.graph = GRAPH_WEIGHT()
        self.final = FINAL_WEIGHT()

class NEIGHBOR_BOOST():
    # SCORE_TYPES = ['tfidf', 'lexrank', 'textrank', 'query_base', 'graph_base', 'ner', 'split_para', 'final']
    def __init__(self):
        self.score_type = 'final'
        self.neighbor_type = 'center'
        self.limit_range = 5
        self.relative_range = (2,2)
        self.threshold = None
        self.boost_first = None
        self.boost_last = None

class SINGLE_SUM_FOR_SINGLE():
    def __init__(self):
        self.limit = 20
        self.limit_type = 'num'
        self.score_type = 'final'
        self.to_text = False
        self.limit_o = 'sentence'
        self.has_bleu = False

class SINGLE_SUM_FOR_MULTI():
    def __init__(self):
        self.limit = 20
        self.limit_type = 'num'
        self.score_type = 'final'
        self.to_text = False
        self.limit_o = 'sentence'
        self.has_bleu = False

class MULTI_SUM():
    def __init__(self):
        self.n_sentences = 13
        self.ratio = None

