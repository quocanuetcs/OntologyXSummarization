import numpy as np
from utils.logger import get_logger

logger = get_logger(__file__)
DATA_FOLDER = '../../../data/abstract/seq2seq/train_set/validation_'
MAX_LEN = 2209
ZERO = np.zeros((1, 768))[0]

