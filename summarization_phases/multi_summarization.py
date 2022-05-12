from copy import deepcopy
from json import JSONEncoder
from entities import Answer
from utils.logger import get_logger
from evaluations.rouge_evaluation import RougeScore
from evaluations.statistic_tool import average_rouge
from preprocessing.preprocessing import load_preprocessing
from summarization_phases.sentence_scoring import load_score_by_name, create_sentece_score
from summarization_phases.single_summarization import single_summarization
from utils.similarity import bert_base
from models.mmr.mmr import Mmr_Summary
logger = get_logger(__file__)
import re

class ObjectEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

def merge_multi_to_single(questions):
    merge_questions = dict()
    for question_id, question in questions.items():
        merge_questions[question_id] = deepcopy(question)
        merge_answer_id = str(question_id) + "_AnswerMerge"
        sentence_cnt = 1
        sentences = dict()
        sentences_str = []
        merge_questions[question_id].answers = dict()
        for answer_id, answer in question.answers.items():
            for sentence_id, sentence in answer.sentences.items():
                sentence_object = deepcopy(sentence)
                sentence_object.answer_id = merge_answer_id
                sentence_object.id = str(sentence_cnt)
                sentences[str(sentence_object.id)] = sentence_object
                sentences_str.append(sentence.sentence)
                sentence_cnt += 1

        merge_questions[str(question_id)].answers[merge_answer_id] = Answer(id=merge_answer_id,
                                                       question_id = str(question_id),
                                                       article= " ".join(sentences_str),
                                                       answer_abs_summ=str(question.multi_abs_summ),
                                                       answer_ext_summ = str(question.multi_ext_summ))
        merge_questions[str(question_id)].answers[merge_answer_id].sentences = sentences
    return merge_questions


def sorted_by_pos(sentences, top=None):
    sentences.sort(key=lambda a: a.start_ques_pos * 100000000 + a.start_ans_pos * 10000 + a.start_sen_pos)
    if top is None:
        top = len(sentences)
    return sentences[:top]

def remove_sentences_contain_urls(sentences):
    results = []
    for sentence in sentences:
        if not bool(re.search(r'http\S+|www.\S+', sentence.sentence)):
            results.append(sentence)
    return results

from entities import nlpEnity

def remove_repeat_sentences(sentences):
    THRESHOLD = 0.9
    nlp = nlpEnity().nlp
    docs = dict()
    for sentence in sentences:
        docs[sentence.id] = nlp(sentence.sentence)

    results = []
    for sentence in sentences:
        current_segmentator = docs[sentence.id]
        is_valid = True
        for previous_sentence in results:
            previous_segmentator = docs[previous_sentence.id]
            if previous_segmentator.doc.similarity(current_segmentator.doc) >= THRESHOLD:
                is_valid = False
                break
        if is_valid:
            results.append(sentence)
    return results

def post_processing(sentences):
    selected_sentences = list()
    for sentence in sentences:
        if '?' not in sentence.sentence and 'example' not in sentence.sentence and len(sentence.ners)<25 and len(sentence.tokens)>1:
            selected_sentences.append(sentence)
    selected_sentences = remove_sentences_contain_urls(selected_sentences)
    #selected_sentences = remove_repeat_sentences(selected_sentences)
    summary = ' '.join(sentence.sentence for sentence in selected_sentences)
    summary = re.sub("[\[].*?[\]]", "", summary)
    return summary

if __name__ == '__main__':
    #config
    version = '1'
    type = 'test'
    run_all = True
    #Preprocessing
    questions = load_preprocessing(name='{}_{}'.format(type,version))
    if not(run_all):
        selected_questions = dict()
        for questionID in ['95']:
            selected_questions[questionID] = questions[questionID]
        questions = selected_questions
    from query_expanding import keyword_expanding
    questions = keyword_expanding(questions, ner_weight=2, token_weight=4)

    if run_all:
        #scores = load_score_by_name(name='{}_{}'.format(type,version),questions=questions)
        scores = create_sentece_score(questions=questions)
    else:
        scores = create_sentece_score(questions=questions)

    #Single-Summarization
    questions = single_summarization(questions=questions, sentence_scores=scores)

    #Single-to-Multi
    merge_questions = merge_multi_to_single(questions=questions)

    #MMR
    result = dict()
    for questionID, question in merge_questions.items():
        logger.info("Get summary for question {}".format(question.id))
        ques_result = {}

        sentences = []
        for answerID, answer in question.answers.items():
            for sentenceID, sentence in answer.sentences.items():
                sentences.append(sentence)

        mmr = Mmr_Summary(sim=bert_base, lambta=0.85)
        selected_sentences = mmr(query=question, n_sentences=25, collection_sents=sentences)
        selected_sentences = sorted_by_pos(selected_sentences)
        summary = post_processing(selected_sentences)
        ques_result['question'] = questions[questionID].question
        ques_result['gen_summary'] = summary
        ques_result['ref_summary'] = questions[questionID].multi_ext_summ
        ques_result['rouge'] = RougeScore().get_score(hypothesis=ques_result['gen_summary'],
                                                      reference=ques_result['ref_summary'])
        result[questionID] = ques_result
        if not(run_all):
            print(summary)
            print(questions[questionID].multi_ext_summ)
            print(ques_result['rouge'])
    if run_all:
        score = average_rouge(result, 'gen_summary', 'ref_summary')
        print(score)



