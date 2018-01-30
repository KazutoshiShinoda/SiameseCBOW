import numpy as np
from ..preprocess.load import PathLineDocuments


class TFIDF():
    """Compute TFIDF vector based on a set of sentences or documents.
    
    Args:
        sentences (list): a list of sentences or documents.
                            each element shoud be a list of words.
                            each word should be parsed and stemmed in advance.
    Ex:
        sentences = [['he', 'is', 'the', ...], ['it', 'is', 'unbeliev', ...]]
    """
    def __init__(self, sentences):
        assert isinstance(sentences, list)
        assert isinstance(sentences[0], list)
        
        unique_words = np.unique(np.concatenate(sentences, 0))
        self.token2id = {u: i for i, u in enumerate(unique_words)}
        
        self.n_sen = len(sentences)
        self.n_vocab = len(self.token2id)
        
        tf = self.compute_tf(sentences)
        tfidf = self.compute_tfidf(tf)
        return tfidf

    def compute_tf(self, sentences):
        tf = np.zeros((self.n_sen, self.n_vocab))
        for row, sentence in enumerate(sentences):
            for word in sentence:
                column = self.token2id[word]
                tf[row, column] += 1
        return tf

    def compute_tfidf(self, tf):
        df = np.count_nonzero(tf, 0)
        idf = self.n_sen / df
        tfidf = df * np.log(idf)
        return tfidf
