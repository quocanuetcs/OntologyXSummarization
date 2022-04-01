from copy import deepcopy
from evaluations.rouge_evaluation import RougeScore
from modules.preprocessing import load_preprocessing
from modules.single_summarization import single_summarization
from models.mmr.mmr import Mmr_Summary
from utils.logger import get_logger
from utils.similarity import *
from models.querybase.querybase_score import Answer
from modules.sentence_scoring import load_score_by_name, create_sentece_score
from evaluations.statistic_tool import average_rouge
from config import MULTI_SUM
from neighbor_boost import neighbor_boost
import json

MULTI_SUM = MULTI_SUM()
logger = get_logger(__file__)

def merge_multi_to_single(questions, scores):
    merge_questions = dict()
    merge_scores = dict()
    for question_id, question in questions.items():
        merge_questions[question_id] = deepcopy(question)
        merge_answer_id = str(question_id) + "_AnswerMerge"
        sentence_cnt = 0
        sentences = dict()
        sentences_str = []
        merge_scores[question_id] = dict()
        merge_questions[question_id].answers = dict()

        for answer_id, answer in question.answers.items():
            merge_scores[question_id][merge_answer_id] = dict()
            for sentence_id, sentence in answer.sentences.items():
                sentence.answer_id = merge_answer_id
                sentence.id = str(sentence_cnt)
                sentences[str(sentence_cnt)] = sentence
                sentences_str.append(sentence.sentence)
                merge_scores[question_id][merge_answer_id][str(sentence_id)] = scores[question_id][answer_id][sentence_id]
                sentence_cnt += 1

        merge_questions[question_id].answers[merge_answer_id] = Answer(id=merge_answer_id,
                                                       question_id = question_id,
                                                       article= " ".join(sentences_str),
                                                       section=" ".join(sentences_str),
                                                       answer_abs_summ=question.multi_abs_summ,
                                                       answer_ext_summ = question.multi_ext_summ)
        merge_questions[question_id].answers[merge_answer_id].sentences = sentences
    return merge_questions, merge_scores

def sorted_by_final(sentences, sentence_scores, n_ranks=None):
    sentences.sort(key=lambda a: sentence_scores[a.question_id][a.answer_id][a.id]['final_score'] if sentence_scores[a.question_id][a.answer_id][a.id]['final_score'] is not None else 0)
    if n_ranks is None:
        n_ranks = len(sentences)
    return sentences[:n_ranks - 1]

def multi_summarization(questions, single_sum_scores):
    result = dict()

    merge_questions, merge_scores = merge_multi_to_single(questions=questions, scores=single_sum_scores)
    merge_scores = create_sentece_score(merge_questions)
    summary = single_summarization(questions=merge_questions, sentence_scores=merge_scores, option='multi')

    mmr = Mmr_Summary(sim=bert_base)
    for question_id, question in summary.items():
        ques_result = {}
        logger.info("getting multiple extractive summary of question: {}".format(question_id))

        sentences = []
        for answer_id, answer in question.answers.items():
            for sentence_id, sentence in answer.sentences.items():
                sentences.append(sentence)

        sentence_score_sorted = sorted_by_final(sentences, merge_scores)
        final_summary = mmr(collection_ranks=sentence_score_sorted, n_sentences=MULTI_SUM.n_sentences, ratio=MULTI_SUM.ratio, query=question)
        ques_result['question'] = questions[question_id].question
        ques_result['gen_summary'] = final_summary
        ques_result['ref_summary'] = questions[question_id].multi_ext_summ
        ques_result['rouge'] = RougeScore().get_score(hypothesis=ques_result['gen_summary'],
                                                      reference=ques_result['ref_summary'])
        result[question_id] = ques_result
    return result


if __name__ == '__main__':
    questions = load_preprocessing(name='validation')
    scores = load_score_by_name('validation')
    scores = neighbor_boost(questions, scores)
    single_output = single_summarization(questions=questions, sentence_scores=scores, option='single')
    multi_output = multi_summarization(single_output, scores)
    score = average_rouge(multi_output, 'gen_summary', 'ref_summary')

    RESULT_PATH = '../data/result/'
    name = 'tfidf'
    json.dump(multi_output, open(RESULT_PATH + name + '_document_summary.json', 'w+', encoding='utf-8'), indent=2, sort_keys=True)
    json.dump(score, open(RESULT_PATH  + name + '_score.json', 'w+'), indent=2, sort_keys=True)