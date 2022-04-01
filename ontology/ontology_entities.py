from utils import preprocessing
from entities import NEREntity, CoreNLPEntity, BioBERTEntity
import numpy as np

BERT_MAX_LENGTH = 512
SPACY = NEREntity()
CORE_NLP = CoreNLPEntity()
BIO_BERT = BioBERTEntity()

class Chemical_Disease_Relation:
    def __init__(self, chemical_node, disease_node, score):
        self.chemical_node = chemical_node
        self.disease_node = disease_node
        self.keys = list()
        self.score = score

    def __eq__(self, other):
        if isinstance(other, Chemical_Disease_Relation):
            if self.chemical_node.ID == other.chemical_node.ID and self.disease_node.ID == other.chemical_node.ID:
                return True
            else: return False

    def update_keys(self, keys):
        for key in keys:
            if not np.isnan(key):
                self.keys.append(key)
        return self

    def combine_keys(self, other):
        if self == other:
            for key in other.keys:
                if key not in self.keys:
                    self.keys.append(key)
            return self

    def combine_score(self, other):
        if self == other:
            self.score = max(self.score, other.socre)
        return self

class Node:
    def __init__(self, ID, stem_from,
                 define = None):
        self.ID = ID
        self.stem_from = stem_from
        self.define = define
        self.define_tokens = list()
        #self.define_ber_tokens = list()
        self.define_ners = list()
        self.terms = list()
        self.parents = list()
        self.children = list()
        self.chemical_disease_relations = list()
        self.treeNumbers = list()

    def __eq__(self,other):
        if isinstance(other,Node):
            if self.ID == other.ID:
                return True
            else: return False

    def add_parent(self, parent):
        if parent not in self.parents:
            self.parents.append(parent)
        return self

    def add_child(self, child):
        if child not in self.children:
            self.children.append(child)
        return self

    def add_chemical_disease_relation(self, chemical_disease_relation):
        if chemical_disease_relation not in self.chemical_disease_relations:
            self.chemical_disease_relations.append(chemical_disease_relation)
        else:
            for relation in self.chemical_disease_relations:
                if relation == chemical_disease_relation:
                    relation.combine_keys(chemical_disease_relation)
                    relation.combine_score(chemical_disease_relation)
        return self

    def add_treeNumbers(self, treeNumber):
        if treeNumber not in self.treeNumbers:
            self.treeNumbers.append(treeNumber)
        return self

    def add_terms(self, term):
        if term not in self.terms:
            self.terms.append(term)
        return self

    def normalize_term(self):
        for index, term in enumerate(self.terms):
            self.terms[index] = preprocessing.normalize(term)
        return self

    def prepare_define(self):
        define_nom = preprocessing.normalize(self.define)
        self.define_tokens = preprocessing.tokenize(define_nom)
        # if len(define_nom.split(' ')) > BERT_MAX_LENGTH:
        #     self.define_ber_tokens = preprocessing.tokenize(define_nom)
        # else:
        #     self.define_ber_tokens = BIO_BERT.tokenize(define_nom)
        self.define_ners = SPACY.get_ners(define_nom)
        return self

    def node_normalize(self):
        self.normalize_term()
        self.prepare_define()
        return self

    def extract_json(self, nodeData):
        self.ID = nodeData["ID"]
        self.terms = nodeData["terms"]
        self.define = nodeData["define"]
        self.treeNumbers = nodeData["treeNumbers"]
        self.define_tokens = nodeData["define_tokens"]
        self.define_ners = nodeData["define_ners"]
        self.parents = nodeData["parents"]
        self.children = nodeData["children"]
        return self

if __name__ == '__main__':
    s = 'New abnormal growth of tissue. Malignant neoplasms show a greater degree of anaplasia and have the properties of invasion and metastasis, compared to benign neoplasms.'
    nom = preprocessing.normalize(s)
    print("DONE")







