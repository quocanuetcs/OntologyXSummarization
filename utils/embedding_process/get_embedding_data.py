from utils.embedding_process.text2vec import *
from utils.embedding_process.text2id import *
import json
from utils.logger import get_logger
import numpy as np

logging = get_logger(__file__)
FOLDER_HOLDING_EMBEDDING_DATA = os.path.dirname(os.path.realpath(__file__)) + '/../../data/embedding_data'


class Dictionary:

    def __init__(self, MAX_THREADS, path_to_folder=FOLDER_HOLDING_EMBEDDING_DATA, from_preprocessing_path=None,
                 js=None):
        self.manager = None
        self.MAX_THREADS = MAX_THREADS
        self.folder = path_to_folder
        if not os.path.exists(path_to_folder):
            logging.info('can not find {} folder\ncreating {}'.format(path_to_folder, path_to_folder))

            try:
                os.makedirs(path_to_folder)
            except OSError:
                logging.info("Creation of the directory %s failed" % path_to_folder)
            else:
                logging.info("Successfully created the directory %s" % path_to_folder)
        try:
            self.dictionary = json.load(open(path_to_folder + '/dictionary.json'))
        except FileNotFoundError:
            logging.info(
                "can not find file dictionary.json in {}\ncreating data from these file: {} and json input".format(
                    path_to_folder, str(
                        from_preprocessing_path)))
            A = text2id(path=from_preprocessing_path, js=js)
            A.run()
            A.export(self.folder)
            self.dictionary = A.data
            logging.info('creating embedding data in {}'.format(self.folder))
            self.get_manager().run(self.dictionary, self.folder)

    def get_manager(self):
        if self.manager is None:
            self.manager = Manager(self.MAX_THREADS)
        return self.manager

    def push(self, sentences=[], preprocessing_path=None, preprocessing_js=None):
        sen_not_in = []
        if preprocessing_path is not None or preprocessing_js is not None:
            A = text2id(preprocessing_path, preprocessing_js)
            A.run()
            sentences.extend(A.sen)
        for s in sentences:
            is_in = False
            for data in self.dictionary:
                if s == data[0]:
                    is_in = True
                    break
            if not is_in:
                sen_not_in.append(s)
        l = [i for i in range(len(self.dictionary), len(self.dictionary) + len(sen_not_in))]
        bart = [(a, i) for a, i in zip(sen_not_in, l)]
        for r in bart:
            self.dictionary.append(r)
        json.dump(self.dictionary, open(self.folder + '/dictionary.json', 'w+'), indent=2)
        self.get_manager().run(bart, self.folder)

    def get_embedding_of(self, sentence):
        # if sentence found in dictionary
        for data in self.dictionary:
            if sentence == data[0]:
                try:
                    # if embedding vector found in data
                    rs = np.load(open(self.folder + '/' + str(data[1]) + '.npy', 'rb'))
                    return rs
                except FileNotFoundError:
                    # logging.info(
                    #     '"{}" is invalid!\nembedding vector will automatically return ZERO'.format(sentence))
                    return ZERO

        # if data not found
        l = len(self.dictionary)
        # append to dictionary
        self.dictionary.append((sentence, l))
        # create embedding data
        self.get_manager().run([(sentence, l)], self.folder)
        json.dump(self.dictionary, open(self.folder + '/dictionary.json', 'w+'), indent=2)
        try:
            # wait until finishing create embedding vector
            while True:
                if not self.get_manager().is_alive():
                    break
            rs = np.load(open(self.folder + '/' + str(l) + '.npy', 'rb'))
            return rs
        except FileNotFoundError:
            logging.info('"{}" is invalid!\nembedding vector will automatically return ZERO'.format(sentence))
            return ZERO


if __name__ == 'main':
    dic = Dictionary(os.path.dirname(os.path.realpath(__file__)) + '/../../data/embedding_data',
                     [os.path.dirname(
                         os.path.realpath(__file__)) + '/../../data/preprocessing/preprocessing_v3_section.json',
                      os.path.dirname(
                          os.path.realpath(__file__)) + '/../../data/validation/preprocessing_validation.json',
                      os.path.dirname(os.path.realpath(__file__)) + '/../../data/preprocessing/preprocessing_v3.json']
                     )
    preprocessing_paths = [
        os.path.dirname(os.path.realpath(__file__)) + '/../../data/preprocessing/preprocessing_v3_section.json',
        os.path.dirname(os.path.realpath(__file__)) + '/../../data/validation/preprocessing_validation.json',
        os.path.dirname(os.path.realpath(__file__)) + '/../../data/preprocessing/preprocessing_v3.json',
        os.path.dirname(os.path.realpath(__file__)) + '/../../data/preprocessing/preprocessing_v3_section_0.json',
        os.path.dirname(os.path.realpath(__file__)) + '/../../data/validation/preprocessing_0.json',
        os.path.dirname(os.path.realpath(__file__)) + '/../../data/preprocessing/preprocessing_v3_0.json'
    ]
    for path in preprocessing_paths:
        dic.push(preprocessing_path=path)
