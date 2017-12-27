import argparse
from models.SiameseCBOW import SiameseCBOW
from preprocess.load import load
import os

#########################################################################################
#  Config
#########################################################################################

file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save')

#########################################################################################
#  Hyper-parameters
#########################################################################################

input_dim = 1000 # Vocabulary size
output_dim = 512 # Sentence length
input_length = 100 # Embedding dimension
n_positive = 2 # Number of positice sample
n_negative = 5 # Number of negative sample


def main():
    parser = argparse.ArgumentParser(description='SiameseCBOW')
    parser.add_argument('--file', '-f', default=None,
                        help='file path of source sentences')
    args = parser.parse_args()
    file=args.file
    x, y = load(file)
    
    model = SiameseCBOW(input_dim, output_dim, input_length=input_length, n_positive=n_positive, n_negative=n_negative)
    model.fit(x, y)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    model.save_embedding_vectors(os.path.join(file_path, 'embedding_vectors.pickle'))
    
if __name__=='__main__':
    main()