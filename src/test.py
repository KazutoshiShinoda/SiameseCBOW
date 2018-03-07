import sys
sys.path.append('.')
import argparse
import numpy as np
from preprocess.load import PathLineDocuments
from models.tfidf import TFIDF
from examples.lexrank import LexRank
from config import SimModel
import pickle


def SiameseCBOW(document):
    with open('../save/embedding_vectors.pickle', 'rb') as f:
        embedding_vectors = pickle.load(f)
    with open('../save/token2id.pickle', 'rb') as f:
        token2id = pickle.load(f)
    emb_dim = embedding_vectors.shape[1]
    unk = token2id['<UNK>']
    
    def _get_ave_embed(sen, embedding_vectors, token2id, emb_dim, unk):
        num = len(sen)
        sen = list(map(lambda x: token2id.get(x, unk), sen))
        tmp = np.zeros(emb_dim)
        for x in sen:
            tmp += embedding_vectors[x]
        return tmp / num

    vectors = list(map(lambda x: _get_ave_embed(x, embedding_vectors, token2id, emb_dim, unk), document))
    return vectors

def main():
    parser = argparse.ArgumentParser(description='Single Document Summarization')
    parser.add_argument('--file', '-f', default=None,
                        help='file path of source sentences')
    parser.add_argument('--sim-model', '-s', type=int, default=None,
                       help='how to compute similarity')
    args = parser.parse_args()
    file = args.file
    sim_model = args.sim_model
    ids_documents = PathLineDocuments(file)
    lexrank = LexRank()
    for ids, document in ids_documents:
        if sim_model == SimModel.tfidf:
            vectors = TFIDF(document)
        elif sim_model == SimModel.siamese_cbow:
            vectors = SiameseCBOW(document)
        score, ranking = lexrank.get_ranking(vectors)
        for i, rank in enumerate(ranking):
            print('%3d' % (i + 1), ' '.join(document[rank]), score[i], sep=' | ')

if __name__ == '__main__':
    main()
