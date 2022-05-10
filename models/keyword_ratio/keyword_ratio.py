from utils.data_loader import *
from models.base_model import BaseModel
from utils.normalize import normalize_by_single_doc
from multiprocessing import Pool

def similarity(keyword_1, keyword_2, threshold=0.8):
    tokens_1 = keyword_1.split(' ')
    tokens_2 = keyword_2.split(' ')
    if len(tokens_1) + len(tokens_2) == 0:
        return 0
    valid_1, valid_2 = dict(), dict()
    for i in range(len(tokens_1)):
        for j in range(len(tokens_2)):
            #if word_similarity(tokens_1[i], tokens_2[j]) >= threshold:
            if tokens_1[i] == tokens_2[j]:
                valid_1[i], valid_2[j] = True, True
    return (len(valid_1) + len(valid_2)) / (len(tokens_1) + len(tokens_2))


def ner_ration(sentence, question, ner_threshold=0.4, word_threshold=0.8):
    valid = dict()

    sentence_keywords = list(sentence.ners)
    for token in sentence.tokens:
        if token not in sentence_keywords:
            sentence_keywords.append(token)

    for i in range(len(sentence_keywords)):
        valid[i] = 0
        for key, key_weight in question.keyword_weights.items():
            if similarity(keyword_1=sentence_keywords[i], keyword_2=key, threshold=word_threshold) >= ner_threshold:
                valid[i] = max(valid[i], key_weight)

    if len(valid)==0:
        return 0
    else:
        score = 0
        weight = 0
        for index, index_weight in valid.items():
            if index_weight != 0:
                score = score + index_weight
                weight = weight + index_weight
            else:
                weight = weight + 1
        return score/weight

class NerRatio(BaseModel):
    NER_THRESHOLD = 0.6
    WORD_THRESHOLD = 0.6

    def __init__(self, ner_threshold=None, word_threshold=None):
        super().__init__()
        self.result = dict()
        self.ner_threshold = ner_threshold if ner_threshold is not None else self.NER_THRESHOLD
        self.word_threshold = word_threshold if word_threshold is not None else self.WORD_THRESHOLD

    def question_train(self, ques):
        ques_id = ques.id
        logger.info('Training NER score of question {}'.format(ques_id))
        question_result = dict()

        for ans_id, ans in ques.answers.items():
            max_score = 0
            answer_result = dict()
            for sen_id, sen in ans.sentences.items():
                answer_result[sen_id] = ner_ration(sen, ques,
                                                   ner_threshold=self.ner_threshold,
                                                   word_threshold=self.word_threshold)
                max_score = max(max_score, answer_result[sen_id])
            if max_score != 0:
                for key in answer_result:
                    answer_result[key] /= max_score
            question_result[ans_id] = answer_result
        return question_result

    def train(self, questions):
        super().train(questions)
        ques_list = []
        for ques_id, ques in self.questions.items():
            ques_list.append(ques)

        with Pool(5) as p:
            results = p.map(self.question_train, ques_list)
        # results = []
        # for index in range(len(ques_list)):
        #     results.append(self.question_train(ques_list[index]))
        for index  in range(len(ques_list)):
            self.result[ques_list[index].id] = results[index]
        self.result = normalize_by_single_doc(self.result)
        return self

    def predict_sentence(self, question_id, answer_id, sentence_id):
        return self.result[question_id][answer_id][sentence_id]


