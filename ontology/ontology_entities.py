from utils import preprocessing
import numpy as np
from entities import nlpEnity

nlp = nlpEnity().nlp

class Node:
    def __init__(self, ID, stem_from,
                 define=None):
        self.ID = ID
        self.stem_from = stem_from
        self.define = define
        self.define_tokens = list()
        self.define_ners = list()
        self.terms = list()
        self.parents = list()
        self.children = list()
        self.chemical_related_node = list()
        self.disease_related_node = list()
        self.gene_related_name = list()
        self.treeNumbers = list()
        self.symptoms = list()
        self.category = list()

    def __eq__(self,other):
        if isinstance(other, Node):
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

    def add_chemical_related_node(self, node):
        if node not in self.chemical_related_node:
            self.chemical_related_node.append(node)
        return self

    def add_disease_related_node(self, node):
        if node not in self.disease_related_node:
            self.disease_related_node.append(node)
        return self

    def add_gene_related_name(self, list_name):
        self.gene_related_name = set(self.gene_related_name)
        self.gene_related_name |= set(list_name)
        self.gene_related_name = list(self.gene_related_name)
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

    def token_tagging(self, doc):
        tokens = []
        for token in doc:
            if not(token.is_stop):
                tokens.append(token.lemma_)
        return tokens

    def ner_tagging(self, doc):
        ners = []
        for ent in doc.ents:
            ners.append(ent.lemma_)
        return ners

    def prepare_define(self):
        define_nom = preprocessing.normalize(self.define)
        doc = nlp(define_nom)
        self.define_tokens = self.token_tagging(doc)
        self.define_ners = self.ner_tagging(doc)
        return self

    def node_normalize(self):
        self.normalize_term()
        if self.define is not None:
            self.prepare_define()
        return self

    def extract_json(self, nodeData):
        self.ID = nodeData["ID"]
        if self.stem_from == "":
            self.stem_from  = nodeData["stem_from"]
        self.terms = nodeData["terms"]
        self.define = nodeData["define"]
        self.treeNumbers = nodeData["treeNumbers"]
        self.define_tokens = nodeData["define_tokens"]
        self.define_ners = nodeData["define_ners"]
        self.parents = nodeData["parents"]
        self.children = nodeData["children"]
        if len(self.treeNumbers)>0:
            for treeNum in self.treeNumbers:
                if 'A' in treeNum and 'A' not in self.category:
                    self.category.append('A')
                elif 'B' in treeNum and 'A' not in self.category:
                    self.category.append('B')
                elif 'C' in treeNum and 'A' not in self.category:
                    self.category.append('C')
                elif 'D' in treeNum and 'A' not in self.category:
                    self.category.append('D')
                elif 'E' in treeNum and 'A' not in self.category:
                    self.category.append('E')
                elif 'F' in treeNum and 'A' not in self.category:
                    self.category.append('F')
                elif 'G' in treeNum and 'A' not in self.category:
                    self.category.append('G')
                elif 'H' in treeNum and 'A' not in self.category:
                    self.category.append('H')
                elif 'I' in treeNum and 'A' not in self.category:
                    self.category.append('I')
                elif 'J' in treeNum and 'A' not in self.category:
                    self.category.append('J')
                elif 'K' in treeNum and 'A' not in self.category:
                    self.category.append('K')
                elif 'L' in treeNum and 'A' not in self.category:
                    self.category.append('L')
                elif 'M' in treeNum and 'A' not in self.category:
                    self.category.append('M')
                elif 'N' in treeNum and 'A' not in self.category:
                    self.category.append('N')
                elif 'V' in treeNum and 'A' not in self.category:
                    self.category.append('V')
                elif 'Z' in treeNum and 'A' not in self.category:
                    self.category.append('Z')
                elif 'R' in treeNum and 'A' not in self.category:
                    self.category.append('R')
                elif 'Y' in treeNum and 'A' not in self.category:
                    self.category.append('Y')
                elif 'X' in treeNum and 'A' not in self.sr:
                    self.category.append('X')
        else:
            self.category.append('C')
        return self

if __name__ == '__main__':
    s = 'New abnormal growth of tissue. Malignant neoplasms show a greater degree of anaplasia and have the properties of invasion and metastasis, compared to benign neoplasms.'
    nom = preprocessing.normalize(s)
    print(nom)







