from json import JSONEncoder
import json
from utils.data_loader import QuestionLoader
from utils.similarity import bert_base
from models.tfidf.tfidf_score import TfIdfScore
from models.lexrank.lexrank_score import LexrankScore
from models.querybase.querybase_score import QueryBaseCore
from models.keyword_ratio.keyword_ratio import NerRatio
from models.wRWMD.wRWMD_score import wRWMD
from utils.logger import get_logger
from config import SENTENCE_SCORING
import os


SENTENCE_SCORING = SENTENCE_SCORING()
logger = get_logger(__file__)


class ObjectEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

def get_tfidf_model(questions):
    model = TfIdfScore(tfidf_ratio_threshold=SENTENCE_SCORING.tfidf.ratio_threshold,
                             keywords_boost=SENTENCE_SCORING.tfidf.keywords_boost,
                             keywords_type=SENTENCE_SCORING.tfidf.keywords_type)
    model.train(questions)
    return  model

def get_lexrank_model(questions, threshold=None):
    if threshold is None: threshold =  SENTENCE_SCORING.lexrank.threshold
    model = LexrankScore(threshold=threshold, tf_for_all_question=SENTENCE_SCORING.lexrank.tf_for_all_question)
    model.train(questions)
    return model

def get_query_base_score_model(questions, sim=bert_base):
    model = QueryBaseCore(sim=sim)
    model.train(questions)
    return model

def get_ner_score_model(questions):
    model = NerRatio(ner_threshold=SENTENCE_SCORING.ner.ner_threshold, word_threshold=SENTENCE_SCORING.ner.word_threshold)
    model.train(questions)
    return model

def get_wRWD_model(questions):
    model = wRWMD()
    model.train(questions)
    return model

def create_sentece_score(questions):
    lexrank_model = get_lexrank_model(questions)
    ner_model = get_ner_score_model(questions)
    wRWMD_model = get_wRWD_model(questions)
    tfidf_model = get_tfidf_model(questions)
    query_base_model = get_query_base_score_model(questions, sim=bert_base)

    result = dict()
    for question_id, question in questions.items():
        result[question_id] = dict()
        final_max_score, final_min_score = 0, 1000000000
        for answer_id, answer in question.answers.items():
            if answer_id not in result[question_id]:
                result[question_id][answer_id] = dict()
            for sentence_id, sentence in answer.sentences.items():
                tfidf_score = tfidf_model.predict_sentence(question_id, answer_id, sentence_id)
                lex_score = lexrank_model.predict_sentence(question_id, answer_id, sentence_id)
                query_score = query_base_model.predict_sentence(question_id, answer_id, sentence_id)
                ner_score = ner_model.predict_sentence(question_id, answer_id, sentence_id)
                wRWD_score = wRWMD_model.predict_sentence(question_id, answer_id, sentence_id)
                final_score = tfidf_score*SENTENCE_SCORING.final.tfidf + \
                                lex_score*SENTENCE_SCORING.final.lexrank + \
                                ner_score*SENTENCE_SCORING.final.ner + \
                                wRWD_score*SENTENCE_SCORING.final.wRWMD +\
                                query_score*SENTENCE_SCORING.final.query_based
                final_score = final_score/SENTENCE_SCORING.final.total_weight
                final_max_score = max(final_max_score, final_score)
                final_min_score = min(final_min_score, final_score)
                result[question_id][answer_id][sentence_id] = {
                    'tfidf_score': tfidf_score,
                    'lexrank_score': lex_score,
                    'query_base_score': query_score,
                    'ner_score': ner_score,
                    'wRWMD':wRWD_score,
                    'final_score': final_score
                }
        for answer_id, answer in question.answers.items():
            for sentence_id, sentence in answer.sentences.items():
                if final_max_score==final_min_score:
                    result[question_id][answer_id][sentence_id]['final_score'] = 1
                else:
                    result[question_id][answer_id][sentence_id]['final_score'] = (result[question_id][answer_id][sentence_id]['final_score'] - final_min_score)/(final_max_score-final_min_score)
    return result

def load_score_by_name(name, link=None, questions=None):
    if link is None:
        score_path = os.path.dirname(os.path.realpath(__file__)) + '/../data/ranking/' + name + '_sentence_scores.json'
    else: score_path = link

    if os.path.exists(score_path):
        logger.info('sentence score exists in {}'.format(score_path))
        pass
    else:
        if questions is None:
            loader = QuestionLoader(name=name).read_json('../data/preprocessing/{}_preprocessing.json'.format(name))
            questions = loader.questions

        scores = create_sentece_score(questions)

        with open(score_path, 'w') as file:
            file.write(json.dumps(scores, cls=ObjectEncoder, indent=2, sort_keys=True))
        logger.info('saved sentence score in {}'.format(score_path))

    with open(score_path) as file:
        scores = json.load(file)
    return scores








