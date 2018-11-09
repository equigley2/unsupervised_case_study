import numpy as np
from numpy import linalg
import pandas as pd
import pickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import text

from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]



class NLP_vect(object):

    def __init__(self, df):
        self.df = df
        self.corpus = df['content']
        self.count_vect = None
        self.count_vect_feature_names = None
        self.tfidf_mat = None
        self.tfidf_mat_feature_names = None

    def count_vect_func(self, stop_words = 'english'):

        vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), stop_words = stop_words)
        self.count_vect = vectorizer.fit_transform(self.corpus).toarray()
        self.count_vect_feature_names = vectorizer.get_feature_names()

    def tfidf_mat_func(self):
        vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer())
        self.tfidf_mat = vectorizer.fit_transform(self.corpus).toarray()
        self.tfidf_mat_feature_names = vectorizer.get_feature_names()

    def split_func(self, matrix):
        train, test = train_test_split(matrix, test_size = 0.2)
        return test , train



if __name__ == '__main__':

    corpus = pd.DataFrame([{'content': 'this is something that is something dog dog in a doc'}, {'content': 'check out this document cat cat it is greater'},{'content': 'check out this document cat dog it is greater agiain this should work'}, {'content': 'check asd ffd ou asds asdt this document asdasd it is greater'}])
    nlp = NLP_vect(corpus)
    nlp.count_vect_func()
    nlp.tfidf_mat_func()
    test, train = nlp.split_func(nlp.count_vect)
