def normalize_by_multi_docs(scores):
    if len(scores) == 0:
        return list()
    for question_id, question in scores.items():
        max_score, min_score = 0, 1000000000
        for answer_id, answer in question.items():
            for sentence_id, sentence in answer.items():
                max_score = max(max_score, scores[question_id][answer_id][sentence_id])
                min_score = min(min_score, scores[question_id][answer_id][sentence_id])
        for answer_id, answer in question.items():
            for sentence_id, sentence in answer.items():
                if max_score == min_score:
                    scores[question_id][answer_id][sentence_id] = 1
                else:
                    score = scores[question_id][answer_id][sentence_id]
                    scores[question_id][answer_id][sentence_id] = (score - min_score) / (max_score - min_score)
    return scores


def normalize_by_single_doc(scores):
    if len(scores) == 0:
        return dict()
    for question_id, question in scores.items():
        for answer_id, answer in question.items():
            max_score, min_score = 0, 1000000000
            for sentence_id, sentence in answer.items():
                max_score = max(max_score, scores[question_id][answer_id][sentence_id])
                min_score = min(min_score, scores[question_id][answer_id][sentence_id])
            for sentence_id, sentence in answer.items():
                if max_score == min_score:
                    scores[question_id][answer_id][sentence_id] = 1
                else:
                    score = scores[question_id][answer_id][sentence_id]
                    scores[question_id][answer_id][sentence_id] = (score - min_score) / (max_score - min_score)
    return scores
