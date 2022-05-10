import pandas as pd
from tqdm import tqdm
from Preprocessing.preprocessing import load_preprocessing
from ontology.constructing import ontology_intergrating
from difflib import SequenceMatcher, get_close_matches
from utils.logger import get_logger
from utils.preprocessing import remove_whitespaces
from ontology.wordnet_query import query_wordnet
logger = get_logger(__file__)


turn_debug = False

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

#Ít nhất 2 trên 3 từ giống nhau trên 80%
def infer_nodeID_from_ner(ontology_node_dict ,ner, word_threshold=0.8, ner_threshold=0.7):
    node_dict = dict()
    node_weight = dict()
    for nodeID, node in ontology_node_dict.items():
        close_words = get_close_matches(ner, node.terms, n=1, cutoff=0.6)
        if len(close_words)>0:
            similarity_score = similarity(ner_1=close_words[0], ner_2=ner, threshold=word_threshold)
            if similarity_score >= ner_threshold:
                if turn_debug: logger.info("Ontology Find {} from {} with close word {}".format(node.ID, ner, close_words[0]))
                node_dict[node.ID] = node
                node_weight[node.ID] = similarity_score
    return node_dict, node_weight

def traveling_node(node, weight, ner_set, weight_dict, decay, traveled_node_ID):
    lower_weight = 1
    traveled_node_ID.append(node.ID)
    if weight<lower_weight: return ner_set, weight_dict, traveled_node_ID
    for term in node.terms:
        if "diseases" in term:
            part = remove_whitespaces(term.replace("diseases", ""))
            ner_set.add(part)
            weight_dict[part] = max(weight_dict.get(part, 0), weight)
        elif "disease" in term:
            part = remove_whitespaces(term.replace("disease", ""))
            ner_set.add(part)
            weight_dict[part] = max(weight_dict.get(part, 0), weight)
        else:
            ner_set.add(term)
            weight_dict[term] = max(weight_dict.get(term, 0), weight)

    parent_weight = child_weight = 0
    if len(node.parents)>0:
        parent_weight =  (1 - decay) * (weight / len(node.parents))

    if len(node.children)>0:
        child_weight = (1 - decay) * (weight / len(node.children))

    if parent_weight>lower_weight:
        for parent in node.parents:
            if parent.ID not in traveled_node_ID:
                ner_set, weight_dict, traveled_node_ID = traveling_node(parent, parent_weight, ner_set, weight_dict, decay, traveled_node_ID)

    if child_weight > lower_weight:
        for child in node.children:
            if child.ID not in traveled_node_ID:
                ner_set, weight_dict, traveled_node_ID = traveling_node(child, child_weight, ner_set, weight_dict, decay, traveled_node_ID)

    # for drug in node.chemical_related_node:
    #     if drug.ID not in traveled_node_ID:
    #         ner_set, weight_dict, traveled_node_ID = traveling_node(drug, weight, ner_set, weight_dict, 1, traveled_node_ID)
    #
    # for disease in node.disease_related_node:
    #     if disease.ID not in traveled_node_ID:
    #         ner_set, weight_dict, traveled_node_ID = traveling_node(disease, weight, ner_set, weight_dict, 1, traveled_node_ID)

    return ner_set, weight_dict, traveled_node_ID

def create_related_ner_from_define(node_dict):
    term_node_dict_A = dict()
    for nodeID, node in node_dict.items():
        if 'D' in node.category:
            for term in node.terms:
                if term not in term_node_dict_A:
                    term_node_dict_A[term] = []
                term_node_dict_A[term].append(nodeID)
    return term_node_dict_A


def expanding_from_ontology(ontology_node_dict, question, ner_weight, related_ner_for_define):
    ner_set = set(question.ners)
    for ner in question.ners:
        question.keyword_weights[ner] = ner_weight

    for ner in question.ners:
        node_dict, node_weight = infer_nodeID_from_ner(ontology_node_dict, ner)
        for nodeID, node in node_dict.items():
            weight = node_weight[nodeID]*ner_weight
            ner_set, question.keyword_weights, traveled_node_ID = traveling_node(node, weight, ner_set, question.keyword_weights, decay=0.3, traveled_node_ID=list())

            for ner in node.define_ners:
                if ner in related_ner_for_define:
                    related_nodesIDs = related_ner_for_define[ner]
                    for related_nodeID in related_nodesIDs:
                        for related_ner in ontology_node_dict[related_nodeID].terms:
                            ner_set.add(related_ner)
                            question.keyword_weights[related_ner] = max(question.keyword_weights.get(related_ner, 0), 6*weight)

            for gene_name in node.gene_related_name:
                ner_set.add(gene_name)
                question.keyword_weights[gene_name] = max(question.keyword_weights.get(gene_name, 0), 2*weight)

            for symp_name in node.symptoms:
                ner_set.add(symp_name)
                question.keyword_weights[symp_name] = max(question.keyword_weights.get(symp_name, 0), weight)

            for drug in node.chemical_related_node:
                for term in drug.terms:
                    if "diseases" in term:
                        part = remove_whitespaces(term.replace("diseases", ""))
                        ner_set.add(part)
                        question.keyword_weights[part] = max(question.keyword_weights.get(part, 0), weight)
                    elif "disease" in term:
                        part = remove_whitespaces(term.replace("disease", ""))
                        ner_set.add(part)
                        question.keyword_weights[part] = max(question.keyword_weights.get(part, 0), weight)
                    else:
                        ner_set.add(term)
                        question.keyword_weights[term] = max(question.keyword_weights.get(term, 0), weight)

            for disease in node.disease_related_node:
                for term in disease.terms:
                    if "diseases" in term:
                        part = remove_whitespaces(term.replace("diseases", ""))
                        ner_set.add(part)
                        question.keyword_weights[part] = max(question.keyword_weights.get(part, 0), weight)
                    elif "disease" in term:
                        part = remove_whitespaces(term.replace("disease", ""))
                        ner_set.add(part)
                        question.keyword_weights[part] = max(question.keyword_weights.get(part, 0), weight)
                    else:
                        ner_set.add(term)
                        question.keyword_weights[term] = max(question.keyword_weights.get(term, 0), weight)

    question.ners_extensions = list()
    for ner in ner_set:
        if ner not in question.ners:
            question.ners_extensions.append(ner)
    return question


def expanding_from_wordnet(question, token_weight):
    output_tokens = set()
    input_tokens = set(question.nouns)
    input_tokens |= set(question.verbs)
    for token in input_tokens:
        if token not in question.extend_from:
            question.extend_from[token] = list()
        question.keyword_weights[token] = max(question.keyword_weights.get(token,0), token_weight)
        extend_token_weight = query_wordnet(token, token_weight=token_weight)
        for extend_token, weight in extend_token_weight.items():
            if extend_token not in input_tokens and len(extend_token.split(' '))==1:
                output_tokens.add(extend_token)
                question.keyword_weights[extend_token] = max(question.keyword_weights.get(extend_token,0), weight)
                question.extend_from[token].append(extend_token)
    question.token_extensions = list(output_tokens)
    return question

def keyword_expanding(questions, ner_weight, token_weight):
    from config import ONTOLOGY_CONFIG
    ONTOLOGY_CONFIG = ONTOLOGY_CONFIG()
    ontology_node_dict = ontology_intergrating(have_mondo=ONTOLOGY_CONFIG.have_mondo, have_symp=ONTOLOGY_CONFIG.have_symp, have_gene=ONTOLOGY_CONFIG.have_gene, have_chemicals_diseases_relation=ONTOLOGY_CONFIG.have_chemicals_diseases_relation)
    logger.info("Term expanding")
    related_ner_for_define = create_related_ner_from_define(node_dict=ontology_node_dict)
    for questionID, question in tqdm(questions.items()):
        if ner_weight != 0: questions[questionID] = expanding_from_ontology(ontology_node_dict, question, ner_weight=ner_weight, related_ner_for_define=related_ner_for_define)
        if token_weight != 0: questions[questionID] = expanding_from_wordnet(question, token_weight=token_weight)
    return questions

