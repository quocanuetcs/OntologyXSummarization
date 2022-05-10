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

    try:
        terms.extend(
                row['A synonym that is recorded for consistency with another source but is a misspelling'].split(
                    "|"))
    except:
        pass

    try:
        terms.extend(row['Synonym to be removed from public release but maintained in edit version as record of external usage'].split("|"))
    except:
        pass

    try:
        terms.extend(row['abbreviation'].split("|"))
    except:
        pass

    return terms

def Mondo_building():
    node_dict_path = os.path.dirname(os.path.realpath(__file__)) + '/Data/Mondo/node_dict.json'
    node_parent_path = os.path.dirname(os.path.realpath(__file__)) + '/Data/Mondo/node_parent_dict.json'
    node_mapping_path = os.path.dirname(os.path.realpath(__file__)) + '/Data/Mondo/node_mapping_dict.json'
    if os.path.exists(node_parent_path) and os.path.exists(node_dict_path):
        logger.info("Load Modo node_dict and node_parent_dict")
        with open(node_parent_path, encoding='utf-8') as file:
            node_parent_dict = json.load(file)

        with open(node_dict_path, encoding='utf-8') as file:
            node_dict_data = json.load(file)

        with open(node_mapping_path, encoding='utf-8') as file:
            mapping_to_mondo = json.load(file)

        node_dict = dict()
        for nodeID, nodeData in node_dict_data.items():
            node_dict[nodeID] = Node(ID=nodeID, stem_from="").extract_json(nodeData)
    else:
        data = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '/Data/Mondo/MONDO.csv', low_memory=False)
        node_dict = dict()
        node_parent_dict = dict()
        mapping_to_mondo = dict()

        for idx in tqdm(data.index):
            row = data.iloc[idx]
            sterm_from, ID = get_node_id(row["Class ID"])
            define = ""
            if len(str(data.loc[idx, "Definitions"]))>5:
                define = data.loc[idx, "Definitions"]
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

            try:
                map_list = row["database_cross_reference"].split('|')
                for value in map_list:
                    if value not in mapping_to_mondo:
                        mapping_to_mondo[value] = list()
                    mapping_to_mondo[value].append(ID)
            except:
                pass

            node = Node(ID=ID, stem_from=sterm_from, define=define)

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
    with open(node_mapping_path, 'w+') as file:
        file.write(json.dumps(mapping_to_mondo, cls=ObjectEncoder, indent=2, sort_keys=True))

    for nodeID, parents in node_parent_dict.items():
        for parent_dict in parents:
            parentID = parent_dict["ID"]
            try:
                node_dict[nodeID].add_parent(node_dict[parentID])
            except:
                logger.info("Can not add parent {} in Monodo Ontology")

            try:
                node_dict[parentID].add_child(node_dict[nodeID])
            except:
                logger.info("Can not add child {} in Monodo Ontology")
    return node_dict, mapping_to_mondo

if __name__ == '__main__':
    node_dict, mapping_to_mondo = Mondo_building()
    node_count = 0
    term_count = 0
    for nodeID, node in node_dict.items():
        node_count+=1
        term_count += len(node.terms)
    print(term_count)
    print("Hello")
