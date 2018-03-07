import numpy as np
from preprocess.load import PathLineDocuments


def TFIDF(sentences):
    """Compute TFIDF vector based on a set of sentences or documents.

    Args:
        sentences (list): a list of sentences or documents.
                            each element shoud be a list of words.
                            each word should be parsed and stemmed in advance.
    Ex:
        sentences = [['he', 'is', 'the', ...], ['it', 'is', 'unbeliev', ...]]
    """
    assert isinstance(sentences, list)
    assert isinstance(sentences[0], list)

    unique_words = np.unique(np.concatenate(sentences, 0))
    token2id = {u: i for i, u in enumerate(unique_words)}

    n_sen = len(sentences)
    n_vocab = len(token2id)

    tf = compute_tf(sentences, token2id, n_sen, n_vocab)
    tfidf = compute_tfidf(tf, n_sen)
    return tfidf

def compute_tf(sentences, token2id, n_sen, n_vocab):
    tf = np.zeros((n_sen, n_vocab))
    for row, sentence in enumerate(sentences):
        for word in sentence:
            column = token2id[word]
            tf[row, column] += 1
    return tf

def compute_tfidf(tf, n_sen):
    df = np.count_nonzero(tf, 0)
    idf = n_sen / df
    tfidf = tf * np.log(idf)
    return tfidf
