from models.ranking import Ranking, QuestionLoader
import json
from utils.logger import *
import numpy as np

logger = get_logger(__file__)
PREPROCESSING_FILE_PATH = '../../data/preprocessing/preprocessing_v3.json'
SCORES_FILE_PATH = '../../data/ranking/scores.json'


def create_scores(questions=None):
    if questions is None:
        loader = QuestionLoader().read_json(PREPROCESSING_FILE_PATH)
        questions = loader.questions
    ranking = Ranking(questions).add_all_score()
    ranking.export_scores()


def load_scores():
    try:
        rs = json.load(open(SCORES_FILE_PATH))
    except Exception as e:
        rs = create_scores()

    x, y = [], []
    for ques_id, ques in rs.items():
        for ans_id, ans in ques.items():
            for sen_id, sen in ans.items():
                m_x = []
                scores = ["query_base_score", "split_para_score", "tfidf_score", "ner_score", "textrank_score",
                          "lexrank_score"]
                for i in range(len(scores)):
                    m_x.append([sen[scores[i]]] if sen[scores[i]] is not None else [0])
                x.append(m_x)
                y.append(sen["precision_rouge_2"])
    x = np.array(x, dtype='float64')
    y = np.array(y, dtype='float64')
    return x, y


if __name__ == '__main__':
    x, y = load_scores()
