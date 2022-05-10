import json
import os
from utils.logger import get_logger
from utils.data_loader import ObjectEncoder
logging = get_logger(__file__)

class text2id:
    def __init__(self, path, js=None):
        self.input = []
        if js is not None:
            self.input = [js]
            self.input = json.loads(json.dumps(self.input, cls=ObjectEncoder))
        if path is not None:
            if isinstance(path, str):
                self.input.append([json.load(open(path))])
            else:
                self.input.extend([json.load(open(p)) for p in path])
        self.data = None
        self.sen = None

    def run(self):
        list_text = []
        for i in self.input:
            for ques_id, ques in i.items():
                logging.info('running in {} ques'.format(ques_id))
                try:
                    list_text.append(ques['question'])
                except:
                    pass

                for ans_id, ans in ques['answers'].items():
                    for sen_id, sen in ans['sentences'].items():
                        list_text.append(sen['sentence'])

        self.data = []
        self.sen = []
        for i in range(len(list_text)):
            self.data.append((list_text[i], i))
            self.sen.append(list_text[i])

    def export(self, path):
        try:
            os.makedirs(path)
        except OSError:
            logging.info("Creation of the directory %s failed" % path)
        else:
            logging.info("Successfully created the directory %s" % path)

        json.dump(self.data, open(path + '/dictionary.json', 'w+'), indent=2)
        logging.info("Successfully exported dictionary in the directory %s/dictionary.json" % path)
