import wn
from config import WORDNET
from utils.preprocessing import remove_stopwords, normalize
from wn.morphy import Morphy

WORDNET = WORDNET()
en = wn.Wordnet('oewn:2021', lemmatizer=Morphy())

def get_sysnets(word_str):
    sysnet_set = set()
    for word in wn.words(word_str):
        sysnet_set |= set(word.synsets())
    return list(sysnet_set)

def get_token_set_from_word_object(word, have_derived_words):
    term_sets = set()
    term_sets.add(word.lemma())
    if WORDNET.word.have_form:
        term_sets |= set(word.forms())
    if have_derived_words:
        for related_word in word.derived_words():
            term_sets |= get_token_set_from_word_object(related_word, have_derived_words=False)
    return term_sets

def traveling_synset(synset, weight, weight_dict, decay, traveled_synset_ID, have_derived_words=True):
    lower_weight = 1
    traveled_synset_ID.append(synset.id)
    if weight<lower_weight: return weight_dict, traveled_synset_ID

    for word in synset.words():
        tokens = get_token_set_from_word_object(word, have_derived_words=have_derived_words)
        tokens = tokens_list_nom(tokens)
        for token in tokens:
            weight_dict[token] = max(weight_dict.get(token, 0), weight)

    hyp_weight = 0
    if len(synset.hypernyms())>0 or len(synset.hyponyms())>0:
        hyp_weight = (1-decay)*(weight/(len(synset.hypernyms())+len(synset.hyponyms())))

    for related_synset in synset.hypernyms():
        weight_dict, traveled_synset_ID = traveling_synset(related_synset, hyp_weight, weight_dict, decay, traveled_synset_ID)

    for related_synset in synset.hyponyms():
        weight_dict, traveled_synset_ID = traveling_synset(related_synset, hyp_weight, weight_dict, decay,traveled_synset_ID)

    return weight_dict, traveled_synset_ID

def tokens_list_nom(tokens):
    tokens_list_nom = remove_stopwords(tokens)
    output_nom_tokens = list()
    for token in tokens_list_nom:
        output_nom_tokens.append(normalize(token))
    return output_nom_tokens

def query_wordnet(word_str, token_weight):
    contain_synsets = get_sysnets(word_str)
    output_token_weight = dict()

    for synset in contain_synsets:
        output_token_weight, traveled_synset_ID = traveling_synset(synset, token_weight, weight_dict=output_token_weight, decay=0.2, traveled_synset_ID=list())
    return output_token_weight

if __name__ == '__main__':
    terms = query_wordnet('treatment', token_weight=2)
    print(terms)






