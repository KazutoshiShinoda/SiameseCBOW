from .load import PathLineDocuments
from numpy import isnan

columns=["sentenceId","category","sectionType","sectionCategory","section4","5","6","7","8","9","10","content"]


class Vocab():
    def __init__(self):
        self.dry=True
        self.token2id={"<UNK>":0}
        self.id2token={}
        
    def build_vocab(self, documents, top_n=None, min_freq=1, keep_vocab=False):
        assert self.dry
        assert isinstance(documents, PathLineDocuments)
        raw_vocab={}
        for ids, doc in documents:
            for sen in doc:
                if isinstance(sen, float):
                    if isnan(sen):
                        continue
                sen = sen.strip().lower().split()
                for word in sen:
                    raw_vocab[word] = raw_vocab.get(word, 0)+1
        self.raw_vocab=raw_vocab
        if top_n:
            stopword = list(map(lambda x: x[0], sorted(self.raw_vocab.items(), key=lambda x: x[1])[-top_n:]))
        else:
            stopword = []
        token2id=self.token2id
        for word in self.raw_vocab.keys():
            if self.raw_vocab[word] > min_freq:
                if not word in token2id and not word in stopword:
                    token2id[word] = len(token2id)
        self.num_vocab=len(token2id)
        self.dry=False
        if not keep_vocab:
            self.raw_vocab={}
        
    def set_id2token(self):
        self.id2token = {v:k for k, v in self.token2id.items()}