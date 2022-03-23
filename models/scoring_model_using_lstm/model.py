from keras.layers import *
from tensorflow.python.keras.utils.np_utils import to_categorical
import numpy as np
from models.base_model import BaseModel

from sklearn.model_selection import train_test_split
from utils.logger import get_logger
from keras.models import *

logger = get_logger(__file__)


class Model:
    model_path = 'calculating_final_score_by_lstm'
    x_shape = (6,1)

    def __init__(self):
        self.model = None
        try:
            self.model = load_model(self.model_path)
        except Exception as e:
            model = Sequential()
            model.add(Input(shape = self.x_shape))
            model.add(Bidirectional(LSTM(64)))
            model.add(Dropout(0.5))
            model.add(Dense(2, activation='sigmoid'))
            logger.info('Complying....')
            model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
            logger.info('Created model successfully!')
            self.model = model

    def train(self, x, y):
        y = [int(i+0.21) for i in y]
        y = np.array(to_categorical(y, num_classes=2))
        logger.info('Training....')
        self.model.fit(x, y)
        self.model.save(self.model_path)
        logger.info('Model saved!')

    def evaluate(self, x, y):
        y = [int(i + 0.21) for i in y]
        y = np.array(to_categorical(y, num_classes=2))
        return self.model.evaluate(x, y)

    def predict(self, x):
        output = None
        if x.shape != self.x_shape:
            output = np.argmax(self.model.predict(x), axis=-1)
        else:
            output = np.argmax(self.model.predict(np.array([x])), axis=-1)[0]
        return output

class Scroing_LSTM(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = Model()
        self.data = None
        self.questions = None

    def train(self, questions, ranking=None):
        super().train()
        self.data = {}
        for ques_id, ques in self.questions.items():
            logger.info('Training question {}'.format(ques_id))
            self.data[ques_id] = {}
            for ans_id, ans in ques.items():
                self.data[ques_id][ans_id] = {}
            rank_of_ques = ranking.ranks[ques_id]
            for rank in rank_of_ques:
                x, y = rank.to_x_y()
                self.data[ques_id][rank.answer_id][rank.sentence_id] = self.model.predict(np.array(x))
        return self

    def predict_sentence(self, ques_id, ans_id, sen_id):
        return self.data[ques_id][ans_id][sen_id]

