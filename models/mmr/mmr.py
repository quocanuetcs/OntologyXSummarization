from entities import SPACY
from models.querybase.prepare_data import create_vocab
import numpy as np
from utils.logger import get_logger
from utils.similarity import *

logger = get_logger(__file__)


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


def get_top_n(query, collection_docs, sim, vocab, lambta, n=None):
    if n is None or n > len(collection_docs):
        n = len(collection_docs)
    result = []
    if vocab is None:
        if not isinstance(collection_docs[0], str):
            m = [sentence.sentence for sentence in collection_docs]
            vocab = create_vocab(m)
        else:
            vocab = create_vocab(collection_docs)
    for i in range(n - 1):
        id = np.argmax(
            [get_score(query=query, doc=d, result=result, lambta=lambta, vocab=vocab, sim=sim) for d in
             collection_docs])
        d = collection_docs[id]
        result.append(d)
        collection_docs.remove(d)
    sorted_by_pos(result)
    if isinstance(result[0], str):
        return result
    return [i.sentence for i in result]

#
# def sorted_by_final(collection_ranks, n_ranks=None):
#     collection_ranks.sort(key=lambda a: a.final_score if a.final_score is not None else 0)
#     if n_ranks is None:
#         n_ranks = len(collection_ranks)
#     return collection_ranks[:n_ranks - 1]


def sorted_by_pos(collection_ranks, n_ranks=None):
    if collection_ranks is None or isinstance(collection_ranks[0], str):
        return collection_ranks
    collection_ranks.sort(key=lambda a: a.start_ques_pos * 100000000 + a.start_ans_pos * 10000 + a.start_sen_pos)
    if n_ranks is None:
        n_ranks = len(collection_ranks)
    return collection_ranks[:n_ranks - 1]


class Mmr_Summary:
    def __init__(self, sim, lambta=None):
        self.sim = sim
        if lambta is None:
            lambta = 0.85
        self.lambta = lambta

    def __call__(self, query=None, collection_docs=None, n_sentences=None, ratio=None, collection_ranks=None,
                 order=None, ):
        if isinstance(collection_docs, str):
            collection_docs = [str(sent) for sent in SPACY.spacy(collection_docs).sents]
        if n_sentences is None:
            if collection_docs is not None:
                n_sentences = len(collection_docs)
            else:
                n_sentences = len(collection_ranks)
        if ratio is not None and isinstance(ratio, float):
            try:
                n_sentences = len(collection_docs) * ratio
            except Exception:
                n_sentences = len(collection_ranks) * ratio
        if collection_ranks is not None:
            collection_docs = []
            for rank in collection_ranks:
                collection_docs.append(rank)
                if query is None:
                    query = rank.question
        return ' '.join(get_top_n(
            query=query,
            collection_docs=collection_docs,
            vocab=None,
            lambta=self.lambta,
            n=n_sentences,
            sim=self.sim))


if __name__ == '__main__':
    collection_docs = """You will be asleep and feel no pain (general anesthesia). The doctor will make a surgical 
    cut (incision)to view the spine. Other surgery, such as a diskectomy, laminectomy, or a foraminotomy, 
    is almost always done first. Spinal fusionmay be done: - On your back or neck over the spine. You will be lying 
    face down. Muscles and tissue will be separated to expose the spine. - On your side, if you are having surgery on 
    your lower back. The surgeon will use tools called retractors to gently separate, hold the soft tissues and blood 
    vessels apart, and have room to work. - With a cut on the front of the neck, toward the side. The surgeon will
    use a graft (such as bone) to hold (or fuse) the bones together permanently. There are several ways of fusing 
    vertebrae together: - Strips of bone graft materialmay be placed over the back part of the spine. - Bone graft 
    material may be placed between the vertebrae. - Special cages may be placed between the vertebrae. These cages 
    are packed with bone graft material. The surgeon may get the bone graft from different places: - From another 
    part of your body (usually around your pelvic bone). This is called an autograft. Your surgeon will make a small 
    cut over your hip and remove some bone from the back of the rim of the pelvis. - From a bone bank. This is called 
    an allograft. - A synthetic bone substitute can also be used. The vertebrae may also fixed together with rods, 
    screws, plates, or cages. They are used to keep the vertebrae from moving until the bone grafts are fully healed. 
    Surgery can take 3to 4 hours, A bone graft can be taken from the person's own healthy bone (this is called an 
    autograft). Or, it can be taken from frozen, donated bone (allograft). In some cases, a manmade (synthetic) bone 
    substitute is used.You will be asleep and feel no pain (general anesthesia).During surgery, the surgeon makes a 
    cut over the bone defect. The bone graft can be taken from areas close to the bone defect or more commonly from 
    the pelvis. The bone graft is shaped and inserted into and around the area. The bone graft can be held in place 
    with pins, plates, or screws, """
    query = "What bone graft materials are used for spinal fusion?"
    mmr = Mmr_Summary(sim=bert_base)
    print(mmr(query, collection_docs, 6))
