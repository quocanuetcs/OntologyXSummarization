from utils.logger import get_logger
import pickle
import json
from json import JSONEncoder
import os
from ontology.MeSH_Constructing import MeSH_building, MeSH_create_node_relation
from ontology.Mondo_Constructing import Mondo_building
from ontology.CTD_relation import create_chemicals_diseases_relation,create_gene_diseases_relation
from ontology.SYMP_Constructing import SYMP_building
from tqdm import tqdm
from difflib import SequenceMatcher, get_close_matches


turn_debug = False
logger = get_logger(__file__)
class ObjectEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

def create_term_node_dict(node_dict):
    dict_path = os.path.dirname(os.path.realpath(__file__)) + '/Data/Intergrated_Ontology/term_node_dict.json'
    logger.info("Load MeSH term_node_dict")
    if os.path.exists(dict_path):
        with open(dict_path, encoding='utf-8') as file:
            term_node_dict = json.load(file)
    else:
        term_node_dict = dict()
        for nodeID, node in node_dict.items():
            for term in node.terms:
                if term not in term_node_dict:
                    term_node_dict[term] = []
                term_node_dict[term].append(nodeID)
        with open(dict_path, 'w+') as file:
            file.write(json.dumps(term_node_dict, cls=ObjectEncoder, indent=2, sort_keys=True))
    return term_node_dict


def word_similarity(word_1, word_2):
    return SequenceMatcher(None, word_1, word_2).ratio()

def similarity(ner_1, ner_2, threshold):
    tokens_1 = ner_1.split(' ')
    tokens_2 = ner_2.split(' ')
    if len(tokens_1) + len(tokens_2) == 0:
        return 0
    valid_1, valid_2 = dict(), dict()
    for i in range(len(tokens_1)):
        for j in range(len(tokens_2)):
            if word_similarity(tokens_1[i], tokens_2[j]) >= threshold:
                valid_1[i], valid_2[j] = True, True
    return (len(valid_1) + len(valid_2)) / (len(tokens_1) + len(tokens_2))


def infer_nodeID_from_ner(ontology_node_dict, ner, word_threshold=0.8, ner_threshold=0.7):
    node_dict = dict()
    for nodeID, node in ontology_node_dict.items():
        close_words = get_close_matches(ner, node.terms, n=1, cutoff=0.6)
        if len(close_words)>0:
            if similarity(ner_1=close_words[0], ner_2=ner, threshold=word_threshold) >= ner_threshold:
                if turn_debug: logger.info("Ontology Find --{}-- from --{}-- with close word --{}--".format(node.ID, ner, close_words[0]))
                node_dict[node.ID] = node
    return [node for nodeID,node in node_dict.items()]

def expanding_from_ontology(ontology_node_dict, ner_list):
    ner_set = set()
    related_nodeIDs = set()
    for ner in ner_list:
        related_nodes = infer_nodeID_from_ner(ontology_node_dict, ner)
        for related_node in related_nodes:
            related_nodeIDs.add(related_node.ID)
            for term in related_node.terms:
                if term not in ner_set:
                    ner_set.add(term)
    return list(ner_set), related_nodeIDs

# def add_symp(node_dict):
#     symp_maping_dict_path = os.path.dirname(os.path.realpath(__file__)) + '/Data/SYMP/diseases_symp_maping_dict.json'
#     #symp_node_maping_dict_path = os.path.dirname(os.path.realpath(__file__)) + '/Data/SYMP/diseases_node_symp_maping_dict.json'
#     if os.path.exists(symp_maping_dict_path):
#         with open(symp_maping_dict_path, encoding='utf-8') as file:
#             symp_maping_dict = json.load(file)
#         for nodeID, node in tqdm(node_dict.items()):
#             if len(symp_maping_dict[nodeID])>0:
#                 node_dict[nodeID].symptoms = symp_maping_dict[nodeID]
#     else:
#         logger.info("Get maping to SYMP")
#         symp_node_maping_dict_path = dict()
#         symp_node_dict = SYMP_building()
#         for nodeID, node in tqdm(node_dict.items()):
#             symps, related_nodeIDs = expanding_from_ontology(ontology_node_dict=symp_node_dict, ner_list=node.define_ners)
#             if len(symps)>0:
#                 node_dict[nodeID].symptoms = symps
#                 symp_node_maping_dict_path[nodeID] = list()
#                 symp_node_maping_dict_path[nodeID].extend(related_nodeIDs)
#         with open(symp_maping_dict_path, 'w+') as file:
#             file.write(json.dumps(symp_node_maping_dict_path, cls=ObjectEncoder, indent=2, sort_keys=True))
#     return node_dict

def add_symp(node_dict):
    symp_node_maping_dict_path = os.path.dirname(os.path.realpath(__file__)) + '/Data/SYMP/diseases_node_symp_maping_dict.json'
    if os.path.exists(symp_node_maping_dict_path):
        with open(symp_node_maping_dict_path, encoding='utf-8') as file:
            symp_node_maping_dict = json.load(file)
        for nodeID, node in tqdm(node_dict.items()):
            try:
                node_dict[nodeID].symptoms = symp_node_maping_dict[nodeID]
            except:
                pass
    else:
        logger.info("Get maping to SYMP")
        symp_node_maping_dict = dict()
        symp_node_dict = SYMP_building()
        for nodeID, node in tqdm(node_dict.items()):
            symps, related_nodeIDs = expanding_from_ontology(ontology_node_dict=symp_node_dict, ner_list=node.define_ners)
            if len(symps)>0:
                node_dict[nodeID].symptoms = symps
                symp_node_maping_dict[nodeID] = list()
                symp_node_maping_dict[nodeID].extend(related_nodeIDs)
        with open(symp_node_maping_dict_path, 'w+') as file:
            file.write(json.dumps(symp_node_maping_dict, cls=ObjectEncoder, indent=2, sort_keys=True))
    return node_dict

def ontology_intergrating(have_mondo, have_symp, have_chemicals_diseases_relation, have_gene):
    node_dict = MeSH_building()
    if have_mondo:
        mondo_node_dict, mapping_to_mondo = Mondo_building()

        logger.info("Ontology Intergrating")
        mondo_remove_ID = list()
        for info, map_list in tqdm(mapping_to_mondo.items()):
            nodeID, term_from = info.split(':')[0], info.split(':')[1]
            if term_from=="MESH" and nodeID in node_dict:
                for mondoID in map_list:
                    logger.info("Merge Mondo mode {} to Mesh node {}".format(mondoID, nodeID))
                    mondo_remove_ID.append(mondoID)
                    mondo_node =  mondo_node_dict[mondoID]
                    mesh_node = node_dict[nodeID]
                    for term in mondo_node.terms:
                        mesh_node.add_terms(term)
                    for parent_node in mondo_node.parents:
                        mesh_node.add_parent(parent_node)
                    for child_node in mondo_node.children:
                        mesh_node.add_child(child_node)
                    if not isinstance(mesh_node.stem_from, list):
                        mesh_node.stem_from = [mesh_node.stem_from]
                    mesh_node.stem_from.append("MONDO")
                    mesh_node.define = ' '.join([mesh_node.define, mondo_node.define])
                    for token in mondo_node.define_tokens:
                        if token not in mesh_node.define_tokens:
                            mesh_node.define_tokens.append(token)
                    for ner in mondo_node.define_tokens:
                        if ner not in mesh_node.define_ners:
                            mesh_node.define_ners.append(ner)
                    node_dict[nodeID] = mesh_node

        for mondoID, mondo_node in mondo_node_dict.items():
            if mondoID not in mondo_remove_ID:
                if mondoID not in node_dict:
                    node_dict[mondoID] = mondo_node
                else:
                    logger.info("ID {} is exist".format(mondoID))
        if have_chemicals_diseases_relation: node_dict = create_chemicals_diseases_relation(node_dict, mapping_to_mondo)
        if have_gene: node_dict = create_gene_diseases_relation(node_dict, mapping_to_mondo)
        if have_symp: node_dict = add_symp(node_dict)
    else:
        if have_chemicals_diseases_relation:
             node_dict = create_chemicals_diseases_relation(node_dict)
        if have_gene: node_dict = create_gene_diseases_relation(node_dict)
        if have_symp: node_dict = add_symp(node_dict)
    return node_dict


if __name__ == '__main__':
    node_dict_symp = SYMP_building()
    node_dict = ontology_intergrating(have_mondo=True, have_symp=True, have_chemicals_diseases_relation=True, have_gene=True)
    count_term = 0
    count_par_child_relation = 0
    count_disease_drug_relation = 0
    count_disease_gene_relation = 0
    count_disease_symp = 0
    syms = set()
    syms_term = []
    genes = set()
    for nodeID, node in node_dict.items():
        count_term = count_term + len(node.terms)
        count_par_child_relation = count_par_child_relation + len(node.parents)
        count_disease_drug_relation = count_disease_drug_relation + len(node.chemical_related_node)
        if len(node.gene_related_name)>0:
            count_disease_gene_relation = count_disease_gene_relation + 1
            genes |= set(node.gene_related_name)
        if len(node.symptoms)>0:
            count_disease_symp = count_disease_symp + len(node.symptoms)
            for sym in node.symptoms:
                if sym not in syms:
                    syms.add(sym)
                    syms_term.extend(node_dict_symp[sym].terms)
    print(len(node_dict))
    print(count_term)
    print(count_par_child_relation)
    print(count_disease_drug_relation)
    print(count_disease_gene_relation)
    print(count_disease_symp)
    print(len(genes))
    print("-------------")
    print(len(syms))
    print(len(syms_term))






