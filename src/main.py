import argparse
from models.SiameseCBOW import SiameseCBOW
from preprocess.load import *
import os
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save')


def main():
    parser = argparse.ArgumentParser(description='SiameseCBOW')
    parser.add_argument('SOURCE', help='source sentence list')
    args = parser.parse_args()
    source=args.SOURCE
    x, y = load(source)
    
    model = SiameseCBOW(input_dim, output_dim, input_length=100, n_potitive=2, n_negative=5)
    model.fit(x, y)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    model.save_embedding_vectors(os.path.join(file_path, 'embedding_vectors.pickle'))
    
if __name__=='__main__':
    main()