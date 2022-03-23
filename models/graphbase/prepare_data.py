from nltk.corpus import stopwords
import json
import os
from utils.data_loader import *
STOPWORDS = stopwords.words('english')
EMBEDDING = BiobertEmbedding()
PATH = os.path.dirname(os.path.realpath(__file__)) 

class Sentence_for_vec:
    def __init__(self, text=''):
        text = text.strip()
        self.data = None
        try:
            self.data = [word_vec.tolist() for word_vec in EMBEDDING.word_vector(text)]
        except:
            m = text.split(',')
            self.data = []
            for m_ in m:
                for x in m_.split(' '):
                    self.data.extend([word_vec.tolist() for word_vec in EMBEDDING.word_vector(x)])


class Paragraph:
    def __init__(self, ans_id, answer):
        self.data = dict()
        self.ans_id = ans_id
        for sen_id, sen in answer.sentences.items():
            self.data[sen_id] = Sentence_for_vec(sen.normalized_sentence).data


def create_file(ques_id, question):
    data = dict()
    print('creating_file_'+str(ques_id)+'.json')
    for ans_id, answer in question.answers.items():
        data[ans_id] = Paragraph(ans_id, answer).data
    json.dump(data, open(PATH+'/../../data/preprocessing/word_vector/' + ques_id + '.json', mode='w'))
    return data


def load_answer(ques_id, question=None):
    if os.path.exists(PATH+'/../../data/preprocessing/word_vector/' + ques_id + '.json'):
        try:
            print('loading_file_'+str(ques_id)+'.json')
            result = json.load(open(PATH+'/../../data/preprocessing/word_vector/' + ques_id + '.json', mode='r'))
            return result
        except:
            print('loading_file_'+str(ques_id)+'.json_failed')
    return create_file(ques_id, question)


def create_vector_word():
    data = QuestionLoader().read_json(PATH+'/../../data/preprocessing/preprocessing.json').questions
    for ques_id, question in data.items():
        if os.path.exists(PATH+'/../../data/preprocessing/word_vector/' + ques_id + '.json'):
            try:
                m = json.load(open(PATH+'/../../data/preprocessing/word_vector/' + ques_id + '.json', mode='r'))
            except:
                create_file(ques_id, question)
        else:
            create_file(ques_id, question)

if __name__=='__main__':
    result = json.load(open(PATH+'/../../data/preprocessing/word_vector/' + '1'+ '.json', mode='r'))