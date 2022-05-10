from utils.similarity import *
from entities import nlpEnity

nlp = nlpEnity().nlp

def get_score(query, doc, result, vocab, sim, lambta):
    if not isinstance(doc, str):
        doc = doc.sentence
    if len(result) == 0:
        return lambta * sim(query, doc, vocab=vocab)
    for i in result:
        if sim(i.sentence , doc, vocab=vocab) >= 0.9:
            return 0
    return lambta * sim(query, doc, vocab=vocab) - (1 - lambta) * np.amax(
        [sim(doc, d.sentence, vocab=vocab) for d in result], axis=0)

def get_top_n(query, collection_sentencs,lambta, n_sentences,  sim=None, vocab=None):
    if sim is None:
        sim = word_weight_base
    if n_sentences > len(collection_sentencs):
        n_sentences = len(collection_sentencs)

    result = []
    for i in range(n_sentences):
        id = np.argmax(
            [get_score(query=query, doc=d, result=result, lambta=lambta, vocab=vocab, sim=sim) for d in
             collection_sentencs])
        d = collection_sentencs[id]
        result.append(d)
        collection_sentencs.remove(d)
    sorted_by_pos(result)
    return [i for i in result]

def sorted_by_pos(sentences, top=None):
    sentences.sort(key=lambda a: a.start_ques_pos * 100000000 + a.start_ans_pos * 10000 + a.start_sen_pos)
    if top is None:
        top = len(sentences)
    return sentences[:top]


class Mmr_Summary:
    def __init__(self, sim=None, lambta=None):
        self.sim = sim
        if lambta is None:
            lambta = 0.65
        self.lambta = lambta

    def __call__(self, query=None, n_sentences=None, ratio=None, collection_sents=None):
        if ratio is not None and isinstance(ratio, float):
                n_sentences = len(collection_sents) * ratio
        if len(collection_sents)<n_sentences: return collection_sents
        return get_top_n(
            query=query,
            collection_sentencs=collection_sents,
            vocab=None,
            lambta=self.lambta,
            n_sentences=n_sentences,
            sim=self.sim)

