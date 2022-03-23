from models.biobert_embedding.embedding import BiobertEmbedding
from threading import Thread
import numpy as np
import time
from utils.logger import get_logger

logging = get_logger(__file__)

Model = BiobertEmbedding()
ZERO = np.zeros((1, 768))[0]


class Embedding_Thread(Thread):
    MODEL = None

    def __init__(self):
        Thread.__init__(self)
        self.name = str(time.time_ns())
        logging.info('Embedding Thread {} created!'.format(self.name))
        if self.MODEL is None:
            self.MODEL = Model

    def run(self, sentence, id, folder_path):
        try:
            vector = self.MODEL.sentence_vector(sentence)
            logging.info('Embedded {} sentence'.format(id))
            np.save(arr=vector, file=open(folder_path + '/' + str(id) + '.npy', 'wb+'))
        except Exception:
            vector = ZERO
            logging.info('Failed embedding {} sentence'.format(id))


class Manager(Thread):
    def __init__(self, size):
        Thread.__init__(self)
        self.name = str(time.time_ns())
        logging.info('Manager Thread {} created!'.format(self.name))
        self.embedding_threads = [Embedding_Thread() for _ in range(size)]

    def run(self, data, folder_path):
        while data is not None and len(data) > 0:
            for thread in self.embedding_threads:
                if not thread.is_alive():
                    thread.run(data[0][0], data[0][1], folder_path)
                    del data[0]
                    break
