import tensorflow as tf
from models.scoring_model_using_lstm.prepare_data import load_scores
from models.scoring_model_using_lstm.model import *

if __name__ == '__main__':
    x, y = load_scores()
    x_train,  x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    md = Model()
    # md.train(x_train, y_train)

    pr =md.predict(x)
    print(pr)
    pass
