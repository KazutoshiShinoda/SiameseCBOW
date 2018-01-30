import nltk
from nltk.stem import PorterStemmer

wnl=nltk.WordNetLemmatizer()
stemmer = PorterStemmer()


def Preprocess(sen):
    """
    Args:
        sen: sentence which is an instance of str.
    """
    sen = sen.lower()
    sen = nltk.word_tokenize(sen)
    sen = list(map(lambda w: stemmer.stem(w), sen))
    # sen = list(map(lambda w: wnl.lemmatize(w), sen))
    return sen
