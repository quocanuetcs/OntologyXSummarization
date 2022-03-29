from models.neighbors.neighbor_boost import NeighborBoost
from utils.logger import get_logger
from config import NEIGHBOR_BOOST

NEIGHBOR_BOOST = NEIGHBOR_BOOST()
logger = get_logger(__file__)

def neighbor_boost(questions, scores):
    model = NeighborBoost(limit_range=NEIGHBOR_BOOST.limit_range,
                           relative_range=NEIGHBOR_BOOST.relative_range,
                           score_type=NEIGHBOR_BOOST.score_type,
                           neighbor_type=NEIGHBOR_BOOST.neighbor_type,
                           threshold=NEIGHBOR_BOOST.threshold,
                           boost_first=NEIGHBOR_BOOST.boost_first,
                           boost_last=NEIGHBOR_BOOST.boost_last)

    score_type = NEIGHBOR_BOOST.score_type
    model.train(questions=questions, input_scores=scores)
    for question_id, question in questions.items():
        for answer_id, answer in question.answers.items():
            for sentence_id, sentence in answer.sentences.items():
                score = model.predict_sentence(question_id, answer_id, sentence_id)
                if score_type == 'final':
                    scores[question_id][answer_id][sentence_id]['final_score'] = score
                elif score_type == 'tfidf':
                    scores[question_id][answer_id][sentence_id]['tfidf_score'] = score
                elif score_type == 'lexrank':
                    scores[question_id][answer_id][sentence_id]['lexrank_score'] = score
                elif score_type == 'query_base':
                    scores[question_id][answer_id][sentence_id]['query_base_score'] = score
                elif score_type == 'graph_base':
                    scores[question_id][answer_id][sentence_id]['graph_score'] = score
                elif score_type == 'ner':
                    scores[question_id][answer_id][sentence_id]['ner_score'] = score
                elif score_type == 'split_para':
                    scores[question_id][answer_id][sentence_id]['split_para_score'] = score
    return scores




