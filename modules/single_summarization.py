from config import SINGLE_SUM_FOR_SINGLE
from config import  SINGLE_SUM_FOR_MULTI

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
    elif score_type == 'graph_base':
        return sentence_score['graph_score']
    elif score_type == 'ner':
        return sentence_score['ner_score']
    elif score_type == 'split_para':
        return sentence_score['split_para_score']
    return None

def single_summarization(questions, sentence_scores, option):
    if option == 'multi':
        score_type = SINGLE_SUM_FOR_SINGLE.score_type
        limit = SINGLE_SUM_FOR_SINGLE.limit
        limit_type = SINGLE_SUM_FOR_SINGLE.limit_type
        to_text = SINGLE_SUM_FOR_SINGLE.to_text
        limit_o = SINGLE_SUM_FOR_SINGLE.limit_o
        has_blue = SINGLE_SUM_FOR_SINGLE.has_bleu
    else:
        score_type = SINGLE_SUM_FOR_MULTI.score_type
        limit = SINGLE_SUM_FOR_MULTI.limit
        limit_type = SINGLE_SUM_FOR_MULTI.limit_type
        to_text = SINGLE_SUM_FOR_MULTI.to_text
        limit_o = SINGLE_SUM_FOR_MULTI.limit_o
        has_blue = SINGLE_SUM_FOR_MULTI.has_bleu

    rs = {}
    for question_id, question in questions.items():
        # Divide into answers
        rs_a = {}
        sen_by_ans = dict()

        for answer_id, answer in question.answers.items():
            if answer_id not in sen_by_ans:
                sen_by_ans[answer_id] = list()
            for sentence_id, sentence in answer.sentences.items():
                sen_by_ans[answer_id].append(sentence)

        for answer_id, sentence_list in sen_by_ans.items():
            if len(sentence_list) == 0:
                continue
            # Filter sentences
            sorted_sentences = sorted(sentence_list, key=lambda obj: get_score(sentence_scores, question_id, answer_id, obj.id, score_type), reverse=True)
            threshold_position = min(limit, len(sorted_sentences))
            if limit_type == 'ratio':
                threshold_position = round(int(limit * len(sorted_sentences)))
            elif limit_type == 'threshold':
                threshold_position = 0
                for i in range(len(sorted_sentences)):
                    if get_score(sentence_scores, question_id, answer_id, sorted_sentences[i].id) < limit:
                        break
                    threshold_position += 1
            threshold_position = max(threshold_position, 1)

            if limit_o == 'sentence':
                sorted_sentences = sorted_sentences[:threshold_position]
            else:
                get_len = 0
                for sentence in sorted_sentences:
                    get_len += len(sentence.tokens)
                if limit_type == 'ratio':
                    get_len = limit * get_len
                else:
                    get_len = limit
                m = []
                n_toks = 0
                for sentence in sorted_sentences:
                    if n_toks + len(sentence.tokens) <= get_len:
                        m.append(sentence)
                        n_toks += len(sentence.tokens)
                    else:
                        break
                sorted_sentences = m
            sorted_sentences = sorted(sorted_sentences, key=lambda obj: int(sentence.id))
            if to_text:
                rs_a[answer_id] = ' '.join([sentence.sentence for sentence in sorted_sentences]).strip()
            else:
                sorted_dict = dict()
                for sentence in sorted_sentences:
                    sorted_dict[sentence.id] = sentence
                questions[question_id].answers[answer_id].sentences = sorted_dict
        rs[question_id] = rs_a

    if to_text:
        return rs
    else:
        return questions

