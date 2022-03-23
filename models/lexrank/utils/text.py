from utils import preprocessing


def tokenize(
    text,
):
    #tokens = text.bert_tokens
    tokens = text.tokens
    tokens = preprocessing.remove_stopwords(tokens)
    tokens = preprocessing.remove_numbers(tokens)
    return tokens
