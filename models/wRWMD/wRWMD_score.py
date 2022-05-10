from utils.data_loader import *
from models.base_model import BaseModel
from utils.normalize import normalize_by_single_doc
from models.wRWMD.prepare_data import cosine, create_vocab
from entities import nlpEnity
from multiprocessing import Pool

nlp = nlpEnity().nlp

def caculate_wRWMD(sen_tokens, ques_tokens,weight_dict, vocab):
    total_score = 0
    total_weight = 0

    for ques_token in ques_tokens:
        max_sim = 0
        for sen_token in sen_tokens:
            try:
                sim_score = cosine(nlp.vocab.get_vector(ques_token), nlp.vocab.get_vector(sen_token))
                max_sim = max(max_sim, sim_score)
            except:
                pass
        keyword_weight = 0
        for keyword, weight in weight_dict.items():
            if (ques_token in keyword):
                keyword_weight = weight
                break
        total_score = total_score + keyword_weight*vocab[ques_token]*max_sim
        total_weight = total_weight + keyword_weight*vocab[ques_token]

    if total_weight!=0:
        return total_score/total_weight
    else:
        return 0

class wRWMD(BaseModel):
    def __init__(self):
        super().__init__()
        self.result = dict()
        self.vocab = dict()

    def question_train(self, ques):
        ques_id = ques.id
        logger.info('Training wRWMD score of question {}'.format(ques_id))
        self.vocab[ques_id] = create_vocab(question=ques)
        question_result = dict()

        ans_tokens = set()
        for ans_id, ans in ques.answers.items():
            for sen_id, sen in ans.sentences.items():
                ans_tokens |= set(sen.tokens)

        ques_tokens = set()
        for token in ques.tokens:
            if token in ans_tokens:
                ques_tokens.add(token)

        if ques.token_extensions is not None:
            for token in ques.token_extensions:
                if token in ans_tokens:
                    ques_tokens.add(token)

        for ans_id, ans in ques.answers.items():
            max_score = 0
            answer_result = dict()
            for sen_id, sen in ans.sentences.items():
                answer_result[sen_id] = caculate_wRWMD(sen.tokens, ques_tokens, weight_dict=ques.keyword_weights,
                                                       vocab=self.vocab[ques_id])
                max_score = max(max_score, answer_result[sen_id])
            if max_score != 0:
                for key in answer_result:
                    answer_result[key] /= max_score
            question_result[ans_id] = answer_result
        return question_result

    def train(self, questions, vocab_dict=None):
        super().train(questions)
        ques_list = []
        for ques_id, ques in self.questions.items():
            ques_list.append(ques)

        with Pool(6) as p:
            results = p.map(self.question_train, ques_list)
        # results = []
        # for ques in ques_list:
        #     results.append(self.question_train(ques))

        for index in range(len(ques_list)):
            self.result[ques_list[index].id] = results[index]
        self.result = normalize_by_single_doc(self.result)
        return self

    def predict_sentence(self, question_id, answer_id, sentence_id):
        return self.result[question_id][answer_id][sentence_id]





