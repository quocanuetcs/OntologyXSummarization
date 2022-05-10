from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def cosine(vec_1, vec_2):
    rs = cosine_similarity([vec_1], [vec_2])[0][0]
    return rs


def custom_tokenize(text):
    return text

def create_vocab_by_idf(question):
    # Create corpus
    corpus = list()
    for answer_id, answer in question.answers.items():
        for sentence_id, sentence in answer.sentences.items():
            tokens = sentence.tokens
            corpus.append(tokens)

    # Create Tf-Idf vectorizer
    vectorizers = TfidfVectorizer(tokenizer=custom_tokenize, lowercase=False)
    vectorizers.fit_transform(corpus)
    vocab = vectorizers.vocabulary_
    rs = {}
    for key in vocab:
        rs[key] = 1 / vectorizers.idf_[vocab[key]]
    return rs


def create_vocab(question):
    iterable_docs = list()
    for ans_id, ans in question.answers.items():
        iterable_docs.extend([' '.join(list(sen.tokens)) for sen_id, sen in ans.sentences.items()])

    victories = TfidfVectorizer(token_pattern='([^\s]+)')
    victories.fit_transform(iterable_docs)
    vocab = victories.vocabulary_
    rs = {}
    for key in vocab:
        rs[key] = 1 / victories.idf_[vocab[key]]
    return rs




