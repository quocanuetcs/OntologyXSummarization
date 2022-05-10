import json
import logging
import os
from json import JSONEncoder

from pandas import DataFrame

from entities import Question
from utils.logger import get_logger

# Disable googleapiclient warnings
##change
# logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)
logger = get_logger(__file__)
# DATASET_PATH = os.path.dirname(os.path.realpath(__file__)) \
#                + '/../data/summarization_datasets/question_driven_answer_summarization_primary_dataset.json'


class ObjectEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


class QuestionLoader:

    def __init__(self, name=None):
        self.file_path = os.path.dirname(
            os.path.realpath(__file__)) + '/../data/preprocessing/' + name + '_preprocessing.json'
        self.questions = dict()
        self.name = name
        self.questions_df = None
        self.answers_df = None
        self.sentences_df = None

    def reload_df(self):
        questions = list()
        answers = list()
        sentences = list()
        for question_id, question in self.questions.items():
            questions.append(question)
            for answer_id, answer in question.answers.items():
                answers.append(answer)
                for sentence_id, sentence in answer.sentences.items():
                    sentences.append(sentence)
        self.questions_df = DataFrame([o.__dict__ for o in questions])
        self.answers_df = DataFrame([o.__dict__ for o in answers])
        self.sentences_df = DataFrame([o.__dict__ for o in sentences])
        return self

    def read_json(self, file_path=None):
        if file_path is None:
            with open(self.file_path, encoding='utf-8') as file:
                data = json.load(file)
        else:
            with open(file_path, encoding='utf-8') as file:
                data = json.load(file)
        for question_id, question in data.items():
            question = Question().extract_question(question_id, question)
            self.questions[question.id] = question
        self.reload_df()
        return self

    def extract_json(self, data=None):
        if os.path.exists(self.file_path):
            self.read_json(self.file_path)
            return self.questions
        for question_id, question in data.items():
            question = Question().extract_question(question_id, question)
            self.questions[question.id] = question
        self.reload_df()
        self.prepare_data()
        return self.export_preprocessing()

    def prepare_data(self):
        for question_id, question in self.questions.items():
            question.prepare_data()
        return self

    def export_preprocessing(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        logger.info('Export preprocessing data to {}'.format(file_path))
        with open(file_path, 'w+') as file:
            file.write(json.dumps(self.questions, cls=ObjectEncoder, indent=2, sort_keys=True))
        return self.questions

