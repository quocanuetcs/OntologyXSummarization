from utils.data_loader import QuestionLoader
from utils.logger import get_logger
from utils.similarity import BERT_DICTIONARY
from utils.to_json import from_in_input_data
import os
logger = get_logger(__file__)

def preprocesing(name):
    input_data = from_in_input_data(name=name.split('_')[0])
    logger.info(name + ' raw loaded!')
    input_data = QuestionLoader(name).extract_json(input_data)
    logger.info(name + ' was preprocessed!')
    BERT_DICTIONARY.push(preprocessing_js=input_data)
    logger.info('embedding {} added!'.format(name))

def load_preprocessing(name):
    path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/data/preprocessing/{}_preprocessing.json'.format(name)
    if os.path.exists(path):
        logger.info('load preprocessing in {}'.format(path))
        pass
    else:
        preprocesing(name=name)
    loader = QuestionLoader(name=name).read_json(path)
    questions = loader.questions
    for question_id, question in questions.items():
        for answer_id, answer in question.answers.items():
            for sentence_id, sentence in answer.sentences.items():
                questions[question_id].answers[answer_id].sentences[sentence_id].start_ques_pos = int(
                    ''.join([i for i in str(question_id) if i in '1234567890']))
                questions[question_id].answers[answer_id].sentences[sentence_id].start_ans_pos = int(
                    ''.join([i for i in str(answer_id) if i in '1234567890']))
                questions[question_id].answers[answer_id].sentences[sentence_id].start_sen_pos = int(
                    ''.join([i for i in str(sentence_id) if i in '1234567890']))
    return questions

if __name__ == '__main__':
    questions = preprocesing(name='test')
    print('Done')

