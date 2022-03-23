from tensorflow.python.keras.utils.np_utils import to_categorical

from models.seq2seq.prepare_data.prepare_data import *
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Model, Input, load_model
from keras.layers import LSTM, Dense, TimeDistributed, Layer, Bidirectional

TAG = {
    1: 'B',
    2: 'I',
    0: 'O'
}
DATA_FOLDER = '../../data/abstract/seq2seq/train_set/validation_'

ZERO = np.zeros((1, 768))[0]


class Seq2Seq:
    X_SHAPE = (2209, 768)
    PATH = 'seq2seq'

    def __init__(self, x_shape=X_SHAPE):
        self.history = None
        self.X_SHAPE = x_shape
        try:
            self.model = load_model(self.PATH)
        except Exception as e:
            print(str(e))
            input = Input(shape=self.X_SHAPE)
            model = Bidirectional(LSTM(units=100, return_sequences=True))(input)  # variational biLSTM
            out = TimeDistributed(Dense(3, activation="softmax"))(model)  # softmax output layer
            self.model = Model(input, out)
            self.model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])


    def train(self, X_tr, y_tr):
        y_tr = np.array([normalize_label(_) for _ in y_tr], dtype='int32')
        y_tr = np.array([to_categorical(i, num_classes=3) for i in y_tr])
        self.history = self.model.fit(X_tr, y_tr, epochs=5)
        self.model.save(self.PATH)
        return self

    def predict(self, input):
        sents = [0 for _ in input]
        if isinstance(input, str):
            X, sents = create_input_from_text(input)
            X = np.array(X)
            Y = self.model.predict(X)
        else:
            Y = self.model.predict(np.array([input]))
        Y = np.argmax(Y, axis=-1)[0]

        table = 'Id\tSentence\tLabel\n'
        for i in range(len(sents)):
            table += '{}\t{}\t{}\n'.format(str(i), str(sents[i]), str(TAG[Y[i]]))
        summarization = get_summarization(sents, Y)

        return table, summarization

    def evaluate(self, x, y):
        y = np.array([to_categorical(i, num_classes=3) for i in y])
        return self.model.evaluate(x, y)



if __name__ == '__main__':
    X, Y = np.load(DATA_FOLDER + 'X2.npy'), np.load(DATA_FOLDER + 'Y2.npy')
    # MAX_LEN = X.shape[1]
    # Y = np.array([normalize_label(_) for _ in Y], dtype='int32')
    # X.shape = (None, 2209, 768)
    # Y.shape = (None, 2209)

    # X_tr, X_te, y_tr, y_te = train_test_split(X, Y, test_size=0.1)
    model = Seq2Seq()
    model.train(X, Y)
    table, _ = model.predict("A bone graft can be taken from the person's own healthy bone (this is called an "
                             "autograft). Or, it can be taken from frozen, donated bone (allograft). In some cases, "
                             "a manmade (synthetic) bone substitute is used.You will be asleep and feel no pain ("
                             "general anesthesia).During surgery, the surgeon makes a cut over the bone defect. The "
                             "bone graft can be taken from areas close to the bone defect or more commonly from the "
                             "pelvis. The bone graft is shaped and inserted into and around the area. The bone graft "
                             "can be held in place with pins, plates, or screws. You will be asleep and feel no pain "
                             "(general anesthesia). The doctor will make a surgical cut (incision)to view the spine. "
                             "Other surgery, such as a diskectomy, laminectomy, or a foraminotomy, is almost always "
                             "done first. Spinal fusionmay be done: - On your back or neck over the spine. You will "
                             "be lying face down. Muscles and tissue will be separated to expose the spine. - On your "
                             "side, if you are having surgery on your lower back. The surgeon will use tools called "
                             "retractors to gently separate, hold the soft tissues and blood vessels apart, "
                             "and have room to work. - With a cut on the front of the neck, toward the side. The "
                             "surgeon will use a graft (such as bone) to hold (or fuse) the bones together "
                             "permanently. There are several ways of fusing vertebrae together: - Strips of bone "
                             "graft materialmay be placed over the back part of the spine. - Bone graft material may "
                             "be placed between the vertebrae. - Special cages may be placed between the vertebrae. "
                             "These cages are packed with bone graft material. The surgeon may get the bone graft "
                             "from different places: - From another part of your body (usually around your pelvic "
                             "bone). This is called an autograft. Your surgeon will make a small cut over your hip "
                             "and remove some bone from the back of the rim of the pelvis. - From a bone bank. This "
                             "is called an allograft. - A synthetic bone substitute can also be used. The vertebrae "
                             "may also fixed together with rods, screws, plates, or cages. They are used to keep the "
                             "vertebrae from moving until the bone grafts are fully healed. Surgery can take 3to 4 "
                             "hours.",
                             )
    print(table)
