import en_core_sci_lg
# import nltk
# import json
from pytextrank import TextRank

from pretrain_models.biobert_embedding.embedding import BiobertEmbedding
from pretrain_models.stanford_corenlp.stanford_corenlp import CoreNLP
from utils import preprocessing


class NEREntity:
    def __init__(self):
        self.spacy = None
        self.has_textrank_pipe = False

    def get_nlp(self):
        if self.spacy is None:
            self.spacy = en_core_sci_lg.load()
        return self.spacy

    def get_ners(self, text):
        if self.spacy is None:
            self.spacy = en_core_sci_lg.load()
        entities = self.spacy(text).ents
        ners = list()
        for entity in entities:
            ners.append(' '.join([token.text for token in entity]))
        return ners

    def get_sents(self, text):
        self.get_nlp()
        l = [str(sen) for sen in self.spacy(text).sents]
        rs = []
        for sen in l:
            if len(sen.split(' ')) > 300:
                a = preprocessing.resplit(sen)
                for i in a:
                    rs.append(i)
            else:
                rs.append(sen)
        return rs

    def add_textrank_pipe(self):
        if not self.has_textrank_pipe:
            self.has_textrank_pipe = True
            tr = TextRank()
            self.get_nlp().add_pipe(tr.PipelineComponent, name="textrank", last=True)
        return self


class CoreNLPEntity:
    def __init__(self):
        self.CORE_NLP = None

    def pos_tag(self, text):
        if self.CORE_NLP is None:
            self.CORE_NLP = CoreNLP()
        return self.CORE_NLP.pos_tag(text)


class BioBERTEntity:
    def __init__(self):
        self.BIO_BERT = None

    def tokenize(self, text):
        if self.BIO_BERT is None:
            self.BIO_BERT = BiobertEmbedding()
        try:
            rs = self.BIO_BERT.word_vector(text).tokens
            return rs
        except:
            return preprocessing.tokenize(text)
        return None


BERT_MAX_LENGTH = 512
SPACY = NEREntity()
CORE_NLP = CoreNLPEntity()
BIO_BERT = BioBERTEntity()


class Sentence:

    def __init__(self, question_id, answer_id, sentence_id, sentence,
                 normalized=None,
                 tokens=None,
                 bert_tokens=None,
                 pos_tags=None,
                 ners=None):
        self.id = sentence_id
        self.sentence = sentence
        self.question_id = question_id
        self.answer_id = answer_id

        self.normalized_sentence = normalized
        self.tokens = tokens
        self.bert_tokens = bert_tokens
        self.pos_tags = pos_tags
        self.ners = ners

        self.start_ques_pos = None
        self.start_ans_pos = None
        self.start_sen_pos = None

    def extract_sentence(self, data):
        self.id = str(data['id'])
        self.sentence = data['sentence']
        self.question_id = data['question_id']
        self.answer_id = data['answer_id']

        self.normalized_sentence = data['normalized_sentence'] if 'normalized_sentence' in data else None
        self.tokens = data['tokens'] if 'tokens' in data else None
        self.bert_tokens = data['bert_tokens'] if 'bert_tokens' in data else None
        self.pos_tags = data['pos_tags'] if 'pos_tags' in data else None
        self.ners = data['ners'] if 'ners' in data else None
        return self

    def normalize(self):
        self.normalized_sentence = preprocessing.normalize(self.sentence)
        return self

    def tokenize(self):
        self.tokens = preprocessing.tokenize(self.normalized_sentence)
        return self

    def tokenize_with_bert(self):
        if len(self.normalized_sentence.split(' ')) > BERT_MAX_LENGTH:
            self.bert_tokens = preprocessing.tokenize(self.normalized_sentence)
        else:
            self.bert_tokens = BIO_BERT.tokenize(self.normalized_sentence)
        return self

    def pos_tagging(self):
        self.pos_tags = CORE_NLP.pos_tag(self.normalized_sentence)
        return self

    def ner_tagging(self):
        self.ners = SPACY.get_ners(self.normalized_sentence)
        return self

    def prepare_data(self):
        self.normalize()
        self.tokenize()
        self.tokenize_with_bert()
        self.pos_tagging()
        self.ner_tagging()
        return self


class Answer:
    def __init__(self,
                 id=None,
                 question_id=None,
                 answer_abs_summ=None,
                 answer_ext_summ=None,
                 section=None,
                 article=None,
                 url=None,
                 rating=None):
        self.id = id
        self.question_id = question_id
        self.answer_abs_summ = answer_abs_summ
        self.answer_ext_summ = answer_ext_summ
        self.section = section
        self.article = article
        self.url = url
        self.rating = rating
        self.sentences = dict()

    def extract_answers(self, question_id, answer_id, data):
        self.id = answer_id
        self.question_id = question_id
        self.answer_abs_summ = data['answer_abs_summ']
        self.answer_ext_summ = data['answer_ext_summ']
        self.section = preprocessing.cleanhtml(data['section'])
        self.article = preprocessing.cleanhtml(data['article'])
        self.url = data['url']
        self.rating = data['rating']

        # Sentences tokenizing with NLTK
        if 'sentences' not in data:
            raw_sentences = SPACY.get_sents(self.article)  # nltk.sent_tokenize(self.article)
            sentence_cnt = 0
            for sentence in raw_sentences:
                if len(sentence.split()) > 500:
                    print(sentence)
                if len(sentence) == 0 or sentence.isspace():
                    continue
                sentence_cnt += 1
                self.sentences[sentence_cnt] = Sentence(question_id, answer_id, sentence_cnt, sentence)
        else:
            for sentence_id, sentence in data['sentences'].items():
                self.sentences[sentence_id] = Sentence(question_id, answer_id, sentence_id, sentence) \
                    .extract_sentence(data['sentences'][sentence_id])

        return self

    def prepare_data(self):
        for sen_id, sen in self.sentences.items():
            sen.prepare_data()
        return self


class Question:

    def __init__(self,
                 id=None,
                 question=None,
                 multi_abs_summ=None,
                 multi_ext_summ=None):
        self.id = id
        self.question = question
        self.multi_abs_summ = multi_abs_summ
        self.multi_ext_summ = multi_ext_summ
        self.answers = dict()

        self.normalized_question = None

        self.tokens = None
        self.bert_tokens = None

        self.pos_tags = None
        self.nouns = None
        self.verbs = None
        self.adjectives = None

        self.ners = None

    def extract_question(self, question_id, data):
        self.id = question_id
        self.question = data['question']
        self.multi_abs_summ = data['multi_abs_summ']
        self.multi_ext_summ = data['multi_ext_summ']
        for answer_id, answer in data['answers'].items():
            self.answers[answer_id] = Answer().extract_answers(question_id, answer_id, answer)

        self.normalized_question = data['normalized_question'] if 'normalized_question' in data else None

        self.tokens = data['tokens'] if 'tokens' in data else None
        self.bert_tokens = data['bert_tokens'] if 'bert_tokens' in data else None

        self.pos_tags = data['pos_tags'] if 'pos_tags' in data else None
        self.nouns = data['nouns'] if 'nouns' in data else None
        self.adjectives = data['adjectives'] if 'adjectives' in data else None
        self.verbs = data['verbs'] if 'verbs' in data else None

        self.ners = data['ners'] if 'ners' in data else None
        return self

    def normalize(self):
        self.normalized_question = preprocessing.normalize(self.question)
        return self

    def tokenize(self):
        self.tokens = preprocessing.tokenize(self.normalized_question)
        return self

    def tokenize_with_bert(self):
        try:
            self.bert_tokens = BIO_BERT.tokenize(self.normalized_question)
        except:
            self.bert_tokens = preprocessing.tokenize(self.normalized_question)
        return self

    def pos_tagging(self):
        self.pos_tags = CORE_NLP.pos_tag(self.normalized_question)
        self.nouns = list()
        self.verbs = list()
        self.adjectives = list()
        for k, v in self.pos_tags:
            if 'NN' in v:
                self.nouns.append(k)
            elif 'VB' in v:
                self.verbs.append(k)
            elif 'JJ' in v:
                self.adjectives.append(k)
        return self

    def ner_tagging(self):
        self.ners = SPACY.get_ners(self.normalized_question)
        return self

    def prepare_data(self):
        self.normalize()
        self.ner_tagging()
        self.pos_tagging()
        self.tokenize_with_bert()
        self.tokenize()
        for ans_id, ans in self.answers.items():
            ans.prepare_data()
        return self

