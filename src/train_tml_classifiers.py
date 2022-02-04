import os
import pandas as pd
import numpy as np
from scipy.stats import randint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from nltk.tokenize import word_tokenize

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
REPLACE_IP_ADDRESS = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')


def text_prepare(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.replace('\n', ' ').lower()  # lowercase text
    text = REPLACE_IP_ADDRESS.sub('', text)
    # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    # delete symbols which are in BAD_SYMBOLS_RE from text
    text = BAD_SYMBOLS_RE.sub('', text)
    # delete stopwords from text
    text = ' '.join([w for w in text.split() if not w in STOPWORDS])
    return text


sentence_labels = pickle.load(open('data.pkl', 'rb'))


df = pd.DataFrame()
df['text'] = [i[0] for i in sentence_labels]
df['labels'] = [i[1] for i in sentence_labels]


def tfidf_features_word(X_train, X_test):
    """
        X_train, X_test — samples        
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test set and return the result

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(
        1, 2), max_df=0.9, min_df=5, token_pattern='(\S+)')

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer.vocabulary_


def tfidf_features_char(X_train, X_test):
    """
        X_train, X_test — samples        
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    tfidf_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(
        1, 5), max_df=0.9, min_df=5, token_pattern='(\S+)')

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer.vocabulary_


def get_f1_score(y_test, predicted, verbose=False):
    if verbose:
        print('F1-score weighted: ',
              f1_score(y_test, predicted, average='weighted'))
        return f1_score(y_test, predicted, average='weighted')


def get_hamming_loss(y_test, predicted, verbose=False):
    if verbose:
        print('Hamming loss: ', hamming_loss(y_test, predicted))
    return hamming_loss(y_test, predicted)


def load_glove_model(File):
    print("Loading Glove Model")
    glove_model = {}
    with open(File, 'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model


def get_glove_embeddings(casefile):
    words = word_tokenize(casefile)
    glove_vals = []
    for word in words:
        try:
            glove_vals.append(glove[word])
        except:
            continue
    if glove_vals == []:
        return np.zeros(300,)
    return np.mean(glove_vals, axis=0)


names = [
    "Linear SVM",
    "RBF SVM",
    "Random Forest",
    "Neural Net",
]

classifiers = [
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    MLPClassifier(alpha=1, max_iter=1000),
]

X, X_test, y, y_test = train_test_split([text_prepare(i) for i in list(
    df['text'])], list(df['labels']), test_size=0.2, train_size=0.8)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, train_size=0.75)

X_train_tfidf, X_test_tfidf, tfidf_vocab = tfidf_feature_word(X_train, X_test)
tfidf_reversed_vocab = {i: word for word, i in tfidf_vocab.items()}


for name, clf in zip(names, classifiers):
    print(name)
    model = OneVsRestClassifier(clf).fit(X_train_tfidf, y_train)

X_train_tfidf, X_test_tfidf, tfidf_vocab = tfidf_feature_char(X_train, X_test)
tfidf_reversed_vocab = {i: word for word, i in tfidf_vocab.items()}

for name, clf in zip(names, classifiers):
    print(name)
    model = OneVsRestClassifier(clf).fit(X_train_tfidf, y_train)

glove = load_glove_model('glove.6B.300d.txt')

X_train_gl, X_test_gl = [get_glove_embeddings(i) for i in X_train], [
    get_glove_embeddings(i) for i in X_test]
for name, clf in zip(names, classifiers):
    print(name)
    model = OneVsRestClassifier(clf).fit(X_train_gl, y_train)
