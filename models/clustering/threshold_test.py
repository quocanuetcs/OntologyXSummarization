from models.clustering.split_model import SplitParaScore
from utils.data_loader import QuestionLoader
from models.ranking import *
import json


def stress_test(step=0.2, times=20, output=None):
    questions = QuestionLoader().read_json('../../data/preprocessing/preprocessing_v2.json').questions
    # questions = dict(itertools.islice(loader.questions.items(), 20))
    rs = dict()
    ranking = Ranking(questions)
    for i in range(1, times + 1):
        threshold = i * step
        ranking.add_split_para_score(threshold=threshold)
        ranking.add_final_score()
        ranking.add_rouge_evaluation()
        rs[threshold] = ranking.summary()['summary'].to_dict()
        print(rs[threshold])
    return rs

if __name__ == '__main__':
    rs = stress_test()
    json.dump(rs, open('threshold_test.json','w+'))
