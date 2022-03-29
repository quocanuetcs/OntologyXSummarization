from json import JSONEncoder
from utils.similarity import word_weight_base, bert_base
from models.tfidf.tfidf_score import TfIdfScore
from models.clustering.split_model import SplitParaScore
from models.lexrank.lexrank_score import LexrankScore
from models.querybase.querybase_score import QueryBaseCore, Answer
from models.graphbase.graph_base_model import GraphScore
from models.ner_ratio.ner_ratio import NerRatio
import os
import json
from utils.logger import get_logger
from utils.data_loader import QuestionLoader
from config import SENTENCE_SCORING

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

def get_lexrank_model(questions):
    model = LexrankScore(threshold=SENTENCE_SCORING.lexrank.threshold, tf_for_all_question=SENTENCE_SCORING.lexrank.tf_for_all_question)
    model.train(questions)
    return model

def get_query_base_score_model(questions, sim=bert_base):
    model = QueryBaseCore(ner_weight=SENTENCE_SCORING.query_base.query_ner_weight, sim=sim)
    model.train(questions)
    return model

def get_ner_score_model(questions):
    model = NerRatio(ner_threshold=SENTENCE_SCORING.ner.ner_threshold, word_threshold=SENTENCE_SCORING.ner.word_threshold)
    model.train(questions)
    return model

def get_split_para_score_model(questions, sim=bert_base):
    model = SplitParaScore(threshold=SENTENCE_SCORING.split.split_threshold, sim=sim)
    model.train(questions)
    return model

def get_graph_score_model(questions, sim=word_weight_base):
    model = GraphScore(ner_weight=SENTENCE_SCORING.graph.ner_weight, sim=sim)
    model.train(questions)
    return model


def create_sentece_score(questions):
    tfidf_model = get_tfidf_model(questions)
    ner_model = get_ner_score_model(questions)
    lexrank_model = get_lexrank_model(questions)
    query_base_model = get_query_base_score_model(questions, sim=bert_base)
    graph_model = get_graph_score_model(questions, sim=word_weight_base)
    split_para_model = get_split_para_score_model(questions)

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
                graph_score = graph_model.predict_sentence(question_id, answer_id, sentence_id)
                ner_score = ner_model.predict_sentence(question_id, answer_id, sentence_id)
                split_score = split_para_model.predict_sentence(question_id, answer_id, sentence_id)
                final_score = tfidf_score*SENTENCE_SCORING.final.tfidf + \
                                lex_score*SENTENCE_SCORING.final.lexrank + \
                                query_score*SENTENCE_SCORING.final.query_base + \
                                graph_score*SENTENCE_SCORING.final.graph + \
                                ner_score*SENTENCE_SCORING.final.ner + \
                                split_score*SENTENCE_SCORING.final.split
                final_score = final_score/SENTENCE_SCORING.final.total_weight
                final_max_score = max(final_max_score, final_score)
                final_min_score = min(final_min_score, final_score)
                result[question_id][answer_id][sentence_id] = {
                    'tfidf_score': tfidf_score,
                    'lexrank_score': lex_score,
                    'query_base_score': query_score,
                    'graph_score': graph_score,
                    'ner_score': ner_score,
                    'split_para_score': split_score,
                    'final_score': final_score
                }
        for answer_id, answer in question.answers.items():
            for sentence_id, sentence in answer.sentences.items():
                if final_max_score==final_min_score:
                    result[question_id][answer_id][sentence_id]['final_score'] = 1
                else:
                    result[question_id][answer_id][sentence_id]['final_score'] = (result[question_id][answer_id][sentence_id]['final_score'] - final_min_score)/(final_max_score-final_min_score)
    return result

def load_score_by_name(name):
    score_path = os.path.dirname(os.path.realpath(__file__)) + '/../data/ranking/' + name + '_sentence_scores.json'
    if os.path.exists(score_path):
        logger.info('sentence score exists in {}'.format(score_path))
        pass
    else:
        loader = QuestionLoader(name=name).read_json('../data/preprocessing/{}_preprocessing.json'.format(name))
        questions = loader.questions

        scores = create_sentece_score(questions)

        with open(score_path, 'w') as file:
            file.write(json.dumps(scores, cls=ObjectEncoder, indent=2, sort_keys=True))
        logger.info('saved sentence score in {}'.format(score_path))

    with open(score_path) as file:
        scores = json.load(file)
    return scores








