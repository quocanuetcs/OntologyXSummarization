from utils.logger import get_logger
from utils.data_loader import QuestionLoader
from models.mmr.run_mmr import export_multiple_summaries
from utils.similarity import *
import json
from evaluations.statistic_tool import average_rouge
from modules.ranking import Ranking
from utils.to_json import from_in_input_data

logger = get_logger(__file__)

RESULT_PATH = '../data/result/'

def test_n_sens(rankings, n_sen):
    a = export_multiple_summaries(rankings, sim=bert_base, opt='from_single_summ', using_mmr='using_mmr',
                                  n_sentences=n_sen,
                                  file_path=RESULT_PATH + str(n_sen) + '_.json')
    x = average_rouge(a, 'gen_summary', 'ref_summary')
    json.dump(x, open(RESULT_PATH + str(n_sen) + '_score.json', 'w+'), indent=2, sort_keys=True)
    return a, x

if __name__ == '__main__':

    data_type = 'VAL'
    name = 'validation'
    input_data = from_in_input_data(type=data_type)
    logger.info('validation raw loaded!')
    input_data = QuestionLoader(name).extract_json(input_data)
    logger.info('data was preprocessed!')
    BERT_DICTIONARY.push(preprocessing_js=input_data)
    logger.info('embedding data added!')
    rankings = Ranking(input_data, name=name)
    rankings.add_all_score()
    rs = {}
    for n_sen in range(8, 20):
        a, x = test_n_sens(rankings, n_sen)
        try:
            rs[n_sen] = x['rouge-2']['f']
        except:
            pass
        logger.info('{} done'.format(str(n_sen)))
    json.dump(rs, open('rs.json', 'w+'), indent=2)




