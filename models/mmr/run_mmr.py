import os
import json
from models.mmr.mmr import Mmr_Summary, sorted_by_final, sorted_by_pos
from evaluations.rouge_evaluation import RougeScore

def export_multiple_summaries(ranking, sim, n_sentences_single=10, n_sentences=13, ratio=None, export=True, file_path=None,
                              opt="from_single_summ",
                              using_mmr='using_mmr'):
    mmr = Mmr_Summary(sim=sim)
    a = dict()
    if file_path is None:
        file_path = os.path.dirname(
            os.path.realpath(
                __file__)) + '/../../data/result/' + ranking.name + '_by_' + opt + '_' + using_mmr + '.json'
    if opt == 'from_single_summ':
        ss = ranking.export_single_summ(limit=n_sentences_single)

    for ques_id, collection_ranks in ranking.ranks.items():
        rs = {}
        # logger.info("getting multiple extractive summary of question: {}".format(ques_id))
        if opt == "from_single_summ":
            collection_ranks = []
            for ans_id, ranks in ss[ques_id].items():
                collection_ranks.extend(ranks)
        if using_mmr == 'not_using_mmr':
            collection_ranks = sorted_by_final(collection_ranks)
            sum = ' '.join([i.sentence.sentence for i in sorted_by_pos(collection_ranks[:n_sentences - 1])])
        else:
            collection_ranks = sorted_by_final(collection_ranks)
            sum = mmr(collection_ranks=collection_ranks, n_sentences=n_sentences, ratio=ratio)
        rs['question'] = ranking.questions[ques_id].question
        rs['gen_summary'] = sum
        rs['ref_summary'] = ranking.questions[ques_id].multi_ext_summ
        rs['rouge'] = RougeScore().get_score(hypothesis=rs['gen_summary'], reference=rs['ref_summary'])
        a[ques_id] = rs
    if export:
        json.dump(a, open(file_path, 'w+', encoding='utf-8'), indent=2, sort_keys=True)
    return a


def tran4submit(js, path):
    with open(path, 'w+', encoding='utf8') as f:
        for ques_id, d in js.items():
            f.write(ques_id + '\t' + d['gen_summary']+'\n')
    f.close()


def evaluate_multiple_ranking(ranking):
    """
    example running mmr
    """
    a = export_multiple_summaries(ranking, opt='from_single_summ', using_mmr='using_mmr')
    a = export_multiple_summaries(ranking, opt='not_from_single_summ', using_mmr='using_mmr')
    a = export_multiple_summaries(ranking, opt='from_single_summ', using_mmr='not_using_mmr')
    a = export_multiple_summaries(ranking, opt='not_from_single_summ', using_mmr='not_using_mmr')
