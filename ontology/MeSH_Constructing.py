import xml.etree.cElementTree as ET
import pandas as pd
from tqdm import tqdm
from ontology.ontology_entities import Node
from utils.logger import get_logger
import pickle
import json
from json import JSONEncoder
import os
from config import MESH_COFIG
MESH_COFIG = MESH_COFIG()

logger = get_logger(__file__)
class ObjectEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

def MeSH_create_node_relation(node_dict, tree_num_dict):
    with open(os.path.dirname(os.path.realpath(__file__)) + '/Data/MeSH/mtrees2021.pkl', 'rb') as f:
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
    tree = ET.parse(os.path.dirname(os.path.realpath(__file__)) + '/Data/MeSH/raw_files/desc2021.xml')
    root = tree.getroot()

    node_dict = dict()
    tree_num_dict = dict()
    for child in tqdm(root):
        ID = child.find('DescriptorUI').text

        defines = []
        for term in child.iter('ScopeNote'):
            defines.append(term.text)

        node = Node(ID=ID,stem_from=["MESH"],define=" ".join(defines))

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

def check_node_choose(node):
    if 'A' in node.category and MESH_COFIG.A:
        return True
    elif 'B' in node.category and MESH_COFIG.B:
        return True
    elif 'C' in node.category and MESH_COFIG.C:
        return True
    elif 'D' in node.category and MESH_COFIG.D:
        return True
    elif 'E' in node.category and MESH_COFIG.E:
        return True
    elif 'F' in node.category and MESH_COFIG.F:
        return True
    elif 'G' in node.category and MESH_COFIG.G:
        return True
    elif 'H' in node.category and MESH_COFIG.H:
        return True
    elif 'I' in node.category and MESH_COFIG.I:
        return True
    elif 'J' in node.category and MESH_COFIG.J:
        return True
    elif 'K' in node.category and MESH_COFIG.K:
        return True
    elif 'L' in node.category and MESH_COFIG.L:
        return True
    elif 'M' in node.category and MESH_COFIG.M:
        return True
    elif 'N' in node.category and MESH_COFIG.N:
        return True
    elif 'V' in node.category and MESH_COFIG.V:
        return True
    elif 'Z' in node.category and MESH_COFIG.Z:
        return True
    elif 'R' in node.category and MESH_COFIG.R:
        return True
    elif 'Y' in node.category and MESH_COFIG.Y:
        return True
    elif 'X' in node.category and MESH_COFIG.X:
        return True
    return False

def MeSH_building():
    tree_num_path = os.path.dirname(os.path.realpath(__file__)) + '/Data/MeSH/tree_num_dict.pkl'
    node_dict_path = os.path.dirname(os.path.realpath(__file__)) + '/Data/MeSH/node_dict.json'
    if os.path.exists(tree_num_path) and os.path.exists(node_dict_path):
        logger.info("Load MeSH node_dict and tree_num_dict")
        with open(tree_num_path, 'rb') as f:
            tree_num_dict = pickle.load(f)

        with open(node_dict_path, encoding='utf-8') as file:
            data = json.load(file)
        node_dict = dict()
        for nodeID, nodeData in data.items():
            node_dict[nodeID] = Node(ID=nodeID, stem_from="MESH").extract_json(nodeData)
    else:
        logger.info("Make MeSH xml detecting")
        node_dict, tree_num_dict = MeSH_xml_detecting()
        with open(tree_num_path, 'wb') as f:
            pickle.dump(tree_num_dict, f)

        with open(node_dict_path, 'w+') as file:
            file.write(json.dumps(node_dict, cls=ObjectEncoder, indent=2, sort_keys=True))

    result_node_dict = dict()
    for nodeID, node in node_dict.items():
        if check_node_choose(node):
            result_node_dict[nodeID] = node

    result_node_dict = MeSH_create_node_relation(node_dict=result_node_dict, tree_num_dict=tree_num_dict)
    return result_node_dict


if __name__ == '__main__':
    mesh_ontology = MeSH_building()
    count_node = 0
    count_relation = 0
    count_term = 0
    for nodeID, node in mesh_ontology.items():
        count_node += 1
        count_relation += len(node.parents)
        count_term += len(node.terms)
    print(count_node, count_relation, count_term)
