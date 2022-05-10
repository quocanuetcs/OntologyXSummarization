from rouge import Rouge
from rouge_score import rouge_scorer


class RougeScore:
    def get_score_2(self, hypothesis, reference):
        return Rouge().get_scores(hypothesis, reference)[0]

    def get_score(self, hypothesis, reference):
        if hypothesis is None:
            hypothesis = ''
        if reference is None:
            reference = ''
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        score = scorer.score(reference, hypothesis)
        return {
            'rouge-1': {
                'f': score['rouge1'].fmeasure,
                'p': score['rouge1'].precision,
                'r': score['rouge1'].recall
            },
            'rouge-2': {
                'f': score['rouge2'].fmeasure,
                'p': score['rouge2'].precision,
                'r': score['rouge2'].recall
            },
            'rouge-l': {
                'f': score['rougeL'].fmeasure,
                'p': score['rougeL'].precision,
                'r': score['rougeL'].recall
            }
        }


def transfer(rouge_scores):
    rs = {}
    for rouge_name, data in rouge_scores.items():
        rouge_name = rouge_name.lower().replace('rouge', 'rouge-')
        rs[rouge_name] = {}
        rs[rouge_name]['r'] = data.recall
        rs[rouge_name]['p'] = data.precision
        rs[rouge_name]['f'] = data.fmeasure
    return rs

