import xml.etree.cElementTree as ET

import pandas as pd
from tqdm import tqdm
from ontology_entities import Node, Chemical_Disease_Relation
from utils.logger import get_logger
import pickle
logger = get_logger(__file__)
import json
from json import JSONEncoder
import os

class ObjectEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

def MeSH_create_node_relation(node_dict, tree_num_dict):
    with open('Data/MeSH/mtrees2021.pkl', 'rb') as f:
        tree_dict = pickle.load(f)

    for nodeID in node_dict:
        node = node_dict[nodeID]
        for tree_number in node.treeNumbers:
            try:
                for child_tree_number in tree_dict[tree_number]:
                    try:
                        childID = tree_num_dict[child_tree_number]
                        child = node_dict[childID]
                        node.add_child(child)
                        child.add_parent(node)
                    except:
                        logger.info("Can't find {}".format(child_tree_number))
            except:
                pass
    return node_dict


def MeSH_xml_detecting():
    tree = ET.parse('Data/MeSH/desc2021.xml')
    root = tree.getroot()

    node_dict = dict()
    tree_num_dict = dict()
    for child in tqdm(root):
        ID = child.find('DescriptorUI').text

        defines = []
        for term in child.iter('ScopeNote'):
            defines.append(term.text)

        node = Node(ID=ID,stem_from="MeSH",define=" ".join(defines))

        for term in child.iter('Term'):
            node.add_terms(term[1].text)

        for tree_num_list in child.iter('TreeNumberList'):
            for tree_num in tree_num_list.iter('TreeNumber'):
                node.add_treeNumbers(tree_num.text)
                if tree_num in tree_num_dict:
                    logger.info("Duplicate tree number in {}{}".format(node.ID,tree_num_dict[tree_num]))
                else:
                    tree_num_dict[tree_num.text] = node.ID

        node = node.node_normalize()
        if node.ID in node_dict:
            logger.info("Dupplicate ID {}".format(node.ID))
            exit()
        else:
            node_dict[node.ID] = node
    return node_dict, tree_num_dict

def MeSH_create_chemicals_diseases_relation(node_dict):
    df_relations = pd.read_csv('Data/CTD/CTD_chemicals_diseases_result.csv')
    count = 0
    for index, row in df_relations.iterrows():
        try:
            chemical_node = node_dict[row['ChemicalID']]
            disease_node = node_dict[row['DiseaseID']]
            relation = Chemical_Disease_Relation(chemical_node=chemical_node, disease_node=disease_node, score=row['InferenceScore'])
            relation.update_keys(keys=[row['DirectEvidence']])
            chemical_node.add_chemical_disease_relation(relation)
            disease_node.add_chemical_disease_relation(relation)
            count += 1
            print(count)
        except:
            pass
    print(count)
    return node_dict


def MeSH_building():
    tree_num_path = 'Data/MeSH/tree_num_dict.pkl'
    node_dict_path = 'Data/MeSH/node_dict.json'
    if os.path.exists(tree_num_path) and os.path.exists(node_dict_path):
        logger.info("Load MeSH node_dict and tree_num_dict")
        with open(tree_num_path, 'rb') as f:
            tree_num_dict = pickle.load(f)

        with open(node_dict_path, encoding='utf-8') as file:
            data = json.load(file)
        node_dict = dict()
        for nodeID, nodeData in data.items():
            node_dict[nodeID] = Node(ID=nodeID, stem_from="MeSH").extract_json(nodeData)
    else:
        logger.info("Make MeSH xml detecting")
        node_dict, tree_num_dict = MeSH_xml_detecting()
        with open(tree_num_path, 'wb') as f:
            pickle.dump(tree_num_dict, f)

        with open(node_dict_path, 'w+') as file:
            file.write(json.dumps(node_dict, cls=ObjectEncoder, indent=2, sort_keys=True))

    logger.info("Make MeSH relation")
    node_dict = MeSH_create_node_relation(node_dict, tree_num_dict)
    node_dict = MeSH_create_chemicals_diseases_relation(node_dict)
    return node_dict

if __name__ == '__main__':
    node_dict = MeSH_building()
    print("DONE")





