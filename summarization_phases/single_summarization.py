from config import SINGLE_SUM_FOR_SINGLE
from config import  SINGLE_SUM_FOR_MULTI
from utils.logger import get_logger

logger = get_logger(__file__)

SINGLE_SUM_FOR_SINGLE = SINGLE_SUM_FOR_SINGLE()
SINGLE_SUM_FOR_MULTI = SINGLE_SUM_FOR_MULTI()

def get_score(input_scores, question_id, answer_id, sentence_id, score_type):
    sentence_score = input_scores[question_id][answer_id][sentence_id]
    if score_type == 'final':
        return sentence_score['final_score']
    elif score_type == 'tfidf':
        return sentence_score['tfidf_score']
    elif score_type == 'lexrank':
        return sentence_score['lexrank_score']
    elif score_type == 'query_base':
        return sentence_score['query_base_score']
    elif score_type == 'ner':
        return sentence_score['ner_score']
    elif score_type == 'wRWMD':
        return sentence_score['wRWMD']
    return None

def single_summarization(questions, sentence_scores):
    score_type = SINGLE_SUM_FOR_SINGLE.score_type
    limit = SINGLE_SUM_FOR_SINGLE.limit
    limit_type = SINGLE_SUM_FOR_SINGLE.limit_type
    threshold = SINGLE_SUM_FOR_SINGLE.threshold

    for question_id, question in questions.items():
        sen_dict = dict()
        for answer_id, answer in question.answers.items():
            if answer_id not in sen_dict:
                sen_dict[answer_id] = list()
            for sentence_id, sentence in answer.sentences.items():
                sen_dict[answer_id].append(sentence)

        for answer_id, sentence_list in sen_dict.items():
            if len(sentence_list) == 0:
                continue
            sorted_sentences = sorted(sentence_list, key=lambda obj: get_score(sentence_scores, question_id, answer_id, obj.id, score_type), reverse=True)
            threshold_position = min(limit, len(sorted_sentences))
            if limit_type == 'ratio':
                threshold_position = round(int(limit * len(sorted_sentences)))
            elif limit_type == 'threshold':
                threshold_position = 0
                for i in range(len(sorted_sentences)):
                    if get_score(sentence_scores, question_id, answer_id, sorted_sentences[i].id, score_type)< limit:
                        break
                    threshold_position += 1
            threshold_position = max(threshold_position, 1)

            choose_sentence = []
            for idx in range(len(sorted_sentences)):
                sentence = sorted_sentences[idx]
                if len(sentence.tokens)<2:
                    continue
                if idx<=threshold_position:
                    choose_sentence.append(sentence)

            choose_sentence = sorted(choose_sentence, key=lambda obj: int(obj.id))

            IDs = []
            for sentence in choose_sentence:
                IDs.append(int(sentence.id))

            add_IDs = IDs.copy()

            rato_length = 3
            for ID in IDs:
                pre = ID
                next = ID
                for index in range(1,rato_length+1):
                    if (ID + index) in IDs:
                        next = ID + index
                    if (ID - index) in IDs:
                        pre = ID - index
                    elif (ID - index)<=0:
                        pre = 1
                for clusterID in range(pre, next+1):
                    if clusterID not in add_IDs:
                        for sentence in sorted_sentences:
                            if int(sentence.id) == clusterID:
                                choose_sentence.append(sentence)
                                add_IDs.append(int(sentence.id))


            choose_sentence = sorted(choose_sentence, key=lambda obj: int(obj.id))

            sorted_dict = dict()
            for sentence in choose_sentence:
                sorted_dict[sentence.id] = sentence

            questions[question_id].answers[answer_id].sentences = sorted_dict
    return questions