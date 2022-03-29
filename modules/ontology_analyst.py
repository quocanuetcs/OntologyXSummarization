import os
import re
import json
from json import JSONEncoder
from utils.data_loader import QuestionLoader

class ObjectEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

if __name__ == '__main__':
    name = 'validation'
    loader = QuestionLoader(name=name).read_json('../data/preprocessing/{}_preprocessing.json'.format(name))
    questions = loader.questions

    result = dict()
    for question_id, question in questions.items():
        result[question_id] = dict()
        result[question_id]['question'] = question.question
        result[question_id]['normalized_question'] = question.normalized_question
        result[question_id]['bert_tokens'] = question.bert_tokens
        result[question_id]['tokens'] = question.tokens
        result[question_id]['ners'] = question.ners
        result[question_id]['pos_tags'] = question.pos_tags
        result[question_id]['verbs'] = question.verbs
        result[question_id]['nouns'] = question.nouns
        result[question_id]['adjectives'] = question.adjectives
        result[question_id]['multi_ext_summ'] = question.multi_ext_summ
        #result[question_id]['multi_abs_summ'] = question.multi_abs_summ
        for answer_id, answer in question.answers.items():
            result[question_id][answer_id] = answer.article

    path = os.path.dirname(os.path.realpath(__file__)) + '/../data/analyst/' + name + '_query_analyst.json'
    with open(path, 'w') as file:
        output = json.dumps(result, cls=ObjectEncoder, indent=2, sort_keys=False)
        output2 = re.sub(r'": \[\s+', '": [', output)
        output3 = re.sub(r'",\s+', '", ', output2)
        output4 = re.sub(r'"\s+\]', '"]', output3)
        file.write(output4)
    print("Done")