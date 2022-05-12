from pretrain_models.stanford_corenlp.stanford_corenlp import CoreNLP
from utils import preprocessing
import spacy
import pysbd

class nlpEnity:
    def __init__(self):
        nlp = spacy.load("en_core_sci_lg")
        self.nlp = nlp

nlp = nlpEnity().nlp

class Sentence:
    def __init__(self, question_id, answer_id, sentence_id, sentence,
                 normalized=None,
                 tokens=None,
                 pos_tags=None,
                 ners=None):
        self.id = sentence_id
        self.sentence = sentence
        self.question_id = question_id
        self.answer_id = answer_id

        self.normalized_sentence = normalized
        self.tokens = tokens
        self.pos_tags = pos_tags
        self.ners = ners
        self.lemma = dict()

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
        self.pos_tags = data['pos_tags'] if 'pos_tags' in data else None
        self.ners = data['ners'] if 'ners' in data else None
        self.lemma = data['lemma'] if 'lemma' in data else None
        return self

    def normalize(self):
        self.normalized_sentence = preprocessing.normalize(self.sentence)
        return self

    def token_tagging(self, doc):
        tokens = []
        for token in doc:
            if not(token.is_stop) and not(token.is_digit):
                tokens.append(token.lemma_)
                self.lemma[token.text] = token.lemma_
        return tokens

    def ner_tagging(self, doc):
        ners = []
        for ent in doc.ents:
            ners.append(ent.lemma_)
            self.lemma[ent.text] = ent.lemma_
        return ners

    def prepare_data(self):
        self.normalize()
        doc = nlp(self.normalized_sentence)
        self.tokens = self.token_tagging(doc)
        self.ners = self.ner_tagging(doc)
        return self

class Answer:
    def __init__(self,
                 id=None,
                 question_id=None,
                 answer_abs_summ=None,
                 answer_ext_summ=None,
                 article=None):
        self.id = id
        self.question_id = question_id
        self.answer_abs_summ = answer_abs_summ
        self.answer_ext_summ = answer_ext_summ
        self.article = article
        self.sentences = dict()

    def extract_answers(self, question_id, answer_id, data):
        self.id = answer_id
        self.question_id = question_id
        self.answer_abs_summ = data['answer_abs_summ']
        self.answer_ext_summ = data['answer_ext_summ']
        self.article = preprocessing.cleanhtml(data['article'])

        if 'sentences' not in data:
            gain_model = pysbd.Segmenter(language="en", clean=True)
            gain_sens = gain_model.segment(self.article)

            new_gain_sens = list()
            index = 0
            while index < len(gain_sens)-1:
                sen = gain_sens[index]
                behind_sen = gain_sens[index+1]

                new_sen = sen
                logic = True
                while sen[-1]=="?" and not(behind_sen[0].isupper()):
                    logic = False
                    new_sen = ''.join([new_sen, behind_sen])
                    index += 1
                    if index < len(gain_sens)-1:
                        sen = gain_sens[index]
                        behind_sen = gain_sens[index + 1]
                    else:
                        break

                if logic:
                    new_gain_sens.append(sen)
                    index += 1
            if index==len(gain_sens)-1: new_gain_sens.append(gain_sens[index])

            sentence_cnt = 0
            for sens in new_gain_sens:
                doc = nlp(sens)
                for sentence in doc.sents:
                    if (sentence is not None) and (len(''.join(sentence.text.split(" ")))>0):
                        sentence_cnt += 1
                        self.sentences[sentence_cnt] = Sentence(question_id, answer_id, sentence_cnt, sentence.text)
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
        self.token_extensions = None

        self.pos_tags = None
        self.nouns = None
        self.verbs = None

        self.ners = None
        self.ners_extensions = None

        self.keyword_weights = dict()
        self.lemma = dict()
        self.extend_from = dict()

    def extract_question(self, question_id, data):
        self.id = question_id
        self.question = data['question']
        self.multi_abs_summ = data['multi_abs_summ']
        self.multi_ext_summ = data['multi_ext_summ']
        for answer_id, answer in data['answers'].items():
            self.answers[answer_id] = Answer().extract_answers(question_id, answer_id, answer)

        self.normalized_question = data['normalized_question'] if 'normalized_question' in data else None
        self.tokens = data['tokens'] if 'tokens' in data else None
        self.token_extensions = data['token_extensions'] if 'token_extensions' in data else None

        self.pos_tags = data['pos_tags'] if 'pos_tags' in data else None
        self.nouns = data['nouns'] if 'nouns' in data else None
        self.verbs = data['verbs'] if 'verbs' in data else None

        self.ners = data['ners'] if 'ners' in data else None
        self.ners_extensions = data['ners_extensions'] if 'ners_extensions' in data else None

        self.lemma = data['lemma'] if 'lemma' in data else dict()
        self.keyword_weights = data['keyword_weights'] if 'keyword_weights' in data else dict()
        self.extend_from = data['extend_from'] if 'extend_from' in data else dict()
        return self

    def normalize(self):
        self.normalized_question = preprocessing.normalize(self.question)
        return self

    def token_tagging(self, doc):
        self.nouns = list()
        self.verbs = list()
        tokens = []
        for token in doc:
            if not(token.is_stop) and not(token.is_digit):
                tokens.append(token.lemma_)
                self.lemma[token.text] = token.lemma_

                if token.pos_ in ['NOUN'] and token.lemma_ not in self.nouns:
                    self.nouns.append(token.lemma_)

                if token.pos_ in ['VERB'] and token.lemma_ not in self.verbs:
                    self.nouns.append(token.lemma_)
        return tokens

    def ner_tagging(self, doc):
        ners = []
        for ent in doc.ents:
            ners.append(ent.lemma_)
            self.lemma[ent.text] = ent.lemma_
        return ners


    def prepare_data(self):
        self.normalize()
        doc = nlp(self.normalized_question)
        self.tokens = self.token_tagging(doc)
        self.ners = self.ner_tagging(doc)
        for ans_id, ans in self.answers.items():
            ans.prepare_data()
        return self

if __name__ == '__main__':
    sentence = 'See this graphic for a quick overview of glaucoma, including how many people it affects, whos at risk, what to do if you have it, and how to learn more.'
    sentence = preprocessing.normalize(sentence)
    doc = nlp(sentence)
    tokens = []
    for token in doc:
        if not (token.is_stop) and not (token.is_digit):
            tokens.append(token.lemma_)