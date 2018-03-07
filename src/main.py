import argparse
from models.SiameseCBOW import SiameseCBOW
from preprocess.load import PathLineDocuments, DataLoader
from preprocess.vocabulary import Vocab
import os
import pickle

#########################################################################################
#  Config
#########################################################################################

file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'save')

#########################################################################################
#  Hyper-parameters
#########################################################################################

input_dim = 1000 # Vocabulary size
output_dim = 512 # Embedding dimension
seq_length = 64 # Sentence length
n_positive = 2 # Number of positice sample
n_negative = 5 # Number of negative sample
batch_size = 32
epochs = 1


def main():
    parser = argparse.ArgumentParser(description='SiameseCBOW')
    parser.add_argument('--file', '-f', default=None,
                        help='file path of source sentences')
    args = parser.parse_args()
    file=args.file
    ids_documents = PathLineDocuments(file)
    vocab = Vocab()
    vocab.build_vocab(ids_documents, min_freq=3)
    input_dim = vocab.num_vocab
    data_loader = DataLoader(ids_documents, batch_size, n_positive, n_negative, seq_length, vocab.token2id)
    if ids_documents.is_counted:
        steps_per_epoch=ids_documents.num_valid_data//batch_size
    else:
        raise ValueError("The number of valid data is not counted.")
    model = SiameseCBOW(input_dim, output_dim, input_length=seq_length, n_positive=n_positive, n_negative=n_negative)
    model.fit_generator(iter(data_loader), steps_per_epoch=80000, epochs=epochs)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    model.save_embedding_vectors(os.path.join(file_path, 'embedding_vectors.pickle'))
    with open(os.path.join(file_path, 'token2id.pickle'), mode='wb') as f:
        pickle.dump(vocab.token2id, f)
    
if __name__=='__main__':
    main()