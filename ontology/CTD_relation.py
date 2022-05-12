import pandas as pd
import os
from utils.logger import get_logger
from json import JSONEncoder
import json
from tqdm import tqdm

logger = get_logger(__file__)
class ObjectEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

def create_chemicals_diseases_relation(node_dict, mapping_to_mondo=None):
    logger.info("Get create_chemicals_diseases_relation")
    df_relations = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) +'/Data/CTD/CTD_chemicals_diseases_result.csv')
    for index, row in tqdm(df_relations.iterrows()):
        try:
            chemical_node = node_dict[row['ChemicalID']]
            disease_infor = row['DiseaseID'].split(':')
            disease_term_from, diseaseID = disease_infor[0], disease_infor[1]
            if disease_term_from == "MESH":
                disease_node = node_dict[diseaseID]
                if disease_term_from not in disease_node.stem_from:
                    #logger.infor("Can not indentify dissease node {}".format(diseaseID))
                    pass
                disease_node.add_chemical_related_node(chemical_node)
                chemical_node.add_disease_related_node(disease_node)
            else:
                if mapping_to_mondo is not None:
                    for mondoID in mapping_to_mondo[':'.join(disease_infor)]:
                        disease_node = node_dict[mondoID]
                        disease_node.add_chemical_related_node(chemical_node)
                        chemical_node.add_disease_related_node(disease_node)
        except:
            pass
    return node_dict

def create_gene_diseases_relation(node_dict, mapping_to_mondo=None):
    logger.info("Get create_gene_diseases_relation")
    dict_path = os.path.dirname(os.path.realpath(__file__)) +'/Data/CTD/CTD_diseases_genes_dict.json'
    with open(dict_path, encoding='utf-8') as file:
        relation_dict = json.load(file)

    for disease_infor, gene_list in tqdm(relation_dict.items()):
        try:
            disease_infor = disease_infor.split(':')
            disease_term_from, diseaseID = disease_infor[0], disease_infor[1]

            if disease_term_from == "MESH":
                disease_node = node_dict[diseaseID]
                if disease_term_from not in disease_node.stem_from:
                    #logger.infor("Can not indentify dissease node {}".format(diseaseID))
                    pass
                disease_node.add_gene_related_name(gene_list)
            else:
                if mapping_to_mondo is not None:
                    for mondoID in mapping_to_mondo[':'.join(disease_infor)]:
                        disease_node = node_dict[mondoID]
                        disease_node.add_gene_related_name(gene_list)
        except:
            pass
    return node_dict