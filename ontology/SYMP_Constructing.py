import pandas as pd
from tqdm import tqdm
from ontology.ontology_entities import Node
from utils.logger import get_logger
from json import JSONEncoder
import os
import json

logger = get_logger(__file__)
class ObjectEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

def get_node_id(s):
    try:
        rs = s.split('/')[-1]
        rs = rs.split('_')
        sterm_from = rs[0]
        ID = rs[1]
    except:
        return
    return [sterm_from], ID


def get_term(row):
    terms = [row['Preferred Label']]
    try:
        terms.extend(row['Synonyms'].split("|"))
    except:
        pass
    return terms

def SYMP_building():
    node_dict_path = os.path.dirname(os.path.realpath(__file__)) + '/Data/SYMP/node_dict.json'
    node_parent_path = os.path.dirname(os.path.realpath(__file__)) + '/Data/SYMP/node_parent_dict.json'
    if os.path.exists(node_parent_path) and os.path.exists(node_dict_path):
        logger.info("Load SYMP node_dict and node_parent_dict")
        with open(node_parent_path, encoding='utf-8') as file:
            node_parent_dict = json.load(file)

        with open(node_dict_path, encoding='utf-8') as file:
            node_dict_data = json.load(file)

        node_dict = dict()
        for nodeID, nodeData in node_dict_data.items():
            node_dict[nodeID] = Node(ID=nodeID, stem_from=["SYMP"]).extract_json(nodeData)
    else:
        data = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '/Data/SYMP/SYMP.csv', low_memory=False)
        node_dict = dict()
        node_parent_dict = dict()

        for idx in tqdm(data.index):
            row = data.iloc[idx]
            sterm_from, ID = get_node_id(row["Class ID"])
            try:
                parents = row["Parents"].split('|')
                for parent in parents:
                    par_stem_from, parID = get_node_id(parent)
                    if ID not in node_parent_dict:
                        node_parent_dict[ID] = list()
                    parent_dict = dict()
                    parent_dict['stem_from'] = par_stem_from
                    parent_dict['ID'] = parID
                    node_parent_dict[ID].append(parent_dict)
            except:
                pass

            node = Node(ID=ID, stem_from=sterm_from)

            terms = get_term(row)
            for term in terms:
                node.add_terms(term)

            node = node.node_normalize()
            if node.ID in node_dict:
                logger.info("Dupplicate ID {}".format(node.ID))
                exit()
            else:
                node_dict[node.ID] = node
    with open(node_parent_path, 'w+') as file:
        file.write(json.dumps(node_parent_dict, cls=ObjectEncoder, indent=2, sort_keys=True))
    with open(node_dict_path, 'w+') as file:
        file.write(json.dumps(node_dict, cls=ObjectEncoder, indent=2, sort_keys=True))

    for nodeID, parents in node_parent_dict.items():
        for parent_dict in parents:
            parentID = parent_dict["ID"]
            try:
                node_dict[nodeID].add_parent(node_dict[parentID])
            except:
                logger.info("Can not add parent {} in SYMP Ontology")

            try:
                node_dict[parentID].add_child(node_dict[nodeID])
            except:
                logger.info("Can not add child {} in SYMP Ontology")
    return node_dict

if __name__ == '__main__':
    node_dict = SYMP_building()
    node_count = 0
    term_count = 0
    for nodeID, node in node_dict.items():
        node_count+=1
        term_count += len(node.terms)
    print(node_count)
    print(term_count)
