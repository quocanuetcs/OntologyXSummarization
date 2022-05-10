from evaluations.rouge_evaluation import RougeScore


def average_rouge(data, key_1='predict_abstract_summ', key_2='actual_abstract_summ'):
    rs = {}
    l = len(data)
    type = ['r', 'p', 'f']
    for ques_id, d in data.items():
        rouge = RougeScore().get_score(d[key_1], d[key_2])
        for e_r in rouge:
            if e_r not in rs:
                rs[e_r] = {}
                for t in type:
                    rs[e_r][t] = 0
            for t in type:
                rs[e_r][t] += rouge[e_r][t] / l
    return rs