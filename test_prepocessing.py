from utils.test_processing import from_validation_in_input_data
from utils.logger import get_logger

logger = get_logger(__file__)

PATH_TO_SOURCE_FOLDER = 'data/raw/Test/'
RESULT_PATH = 'data/result/Test'

if __name__ == '__main__':
    input_data = from_validation_in_input_data(PATH_TO_SOURCE_FOLDER)
    logger.info('data loaded!')