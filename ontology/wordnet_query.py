import wn
from config import WORDNET
WORDNET = WORDNET()
wn = wn.Wordnet('oewn:2021')

def get_sysnets(word_str):
    sysnet_set = set()
    for word in wn.words(word_str):
        sysnet_set |= set(word.synsets())
    return list(sysnet_set)

def get_term_set_from_word_object(word, have_derived_words):
    term_sets = set()
    term_sets.add(word.lemma())
    if WORDNET.word.have_form:
        term_sets |= set(word.forms())
    if have_derived_words:
        for related_word in word.derived_words():
            term_sets |= get_term_set_from_word_object(related_word, have_derived_words=False)
    return term_sets

def query_wordnet(word_str):
    contain_synsets = get_sysnets(word_str)
    output_term_set = set()

    for synset in contain_synsets:
        for word in synset.words():
            output_term_set |= get_term_set_from_word_object(word, have_derived_words=WORDNET.word.have_derived_forms)
        if WORDNET.synset.have_hypernyms_synset:
            for related_synset in synset.hypernyms():
                for word in related_synset.words():
                    output_term_set |= get_term_set_from_word_object(word, have_derived_words=False)
        if WORDNET.synset.have_hyponums_synset:
            for related_synset in synset.hypernyms():
                for word in related_synset.words():
                    output_term_set |= get_term_set_from_word_object(word, have_derived_words=False)
    return list(output_term_set)

if __name__ == '__main__':
    terms = query_wordnet('treatment')
    print(len(terms))
    for term in terms:
        print(term)






