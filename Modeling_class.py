from NLP_class import NLP_vect

import numpy as np
from numpy import linalg
import pandas as pd
import pickle
from numpy.linalg import svd
import matplotlib.pyplot as plt


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import text

from sklearn import decomposition



class Model_UFO(object):

    def __init__(self,NLP_vect_class):
        self.NLP_vect_class = NLP_vect_class
        self.SVD = None
        self.SVD_mat = None
        self.GMM = None
        self.simp_SVD = None
        self.PCA = None

    def SVD_model(self, full_matrices=False, compute_uv=True):
        U, sigma, VT = svd(self.NLP_vect_class.count_vect, full_matrices=full_matrices, compute_uv=compute_uv)
        svd_model = TruncatedSVD(n_components=2)
        self.SVD_mat = svd_model.fit_transform(self.NLP_vect_class.tfidf_mat)
        self.SVD = svd_model
        self.simp_SVD = [U, sigma, VT]

    def scree_SVD(self):
        explained_variance_ratio = self.SVD.explained_variance_ratio_

        # plt.figure(figsize=(6, 6), dpi=250)
        plt.figure(2)
        cum_var = np.cumsum(explained_variance_ratio)
        ax = plt.subplot(111)

        ax.plot(range(len(explained_variance_ratio) + 1), np.insert(cum_var, 0, 0), color = 'r', marker = 'o')
        ax.bar(range(len(explained_variance_ratio)), explained_variance_ratio, alpha = 0.8)

        ax.axhline(0.9, color = 'g', linestyle = "--")
        ax.set_xlabel("Principal Component", fontsize=12)
        ax.set_ylabel("Variance Explained (%)", fontsize=12)

        plt.title("Scree Plot for the Digits Dataset", fontsize=16)
        # plt.show()

    def cluster_SVD(self,X,title='cluster plot'):
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)
        plt.figure(1)
        ax = plt.subplot(111)
        ax.axis('off')
        ax.patch.set_visible(False)
        for i in range(X.shape[0]):
            string1 = np.arange(self.SVD.explained_variance_ratio_.shape[0]+1)
            plt.text(X[i, 0], X[i, 1], str(string1[i]),
                     fontdict={'weight': 'bold', 'size': 12})

        plt.xticks([]), plt.yticks([])
        plt.ylim([-0.1,1.1])
        plt.xlim([-0.1,1.1])

        if title is not None:
            plt.title(title, fontsize=16)

        plt.show()

    def PCA_func(self):
        pca = decomposition.PCA()
        pca.fit(self.NLP_vect_class.tfidf_mat)
        self.PCA = pca
        total_variance = np.sum(pca.explained_variance_)
        cum_variance = np.cumsum(pca.explained_variance_)
        prop_var_expl = cum_variance/total_variance
        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(prop_var_expl, color = 'black', linewidth=2, label='Explained variance')
        ax.axhline(0.9, label='90% goal', linestyle='--', linewidth=1)
        ax.set_ylabel('proportion of explained variance')
        ax.set_xlabel('number of principal components')
        ax.legend()
        plt.show()




if __name__ == '__main__':
    corpus = pd.DataFrame([{'content': 'this is something that is something dog dog in a doc'}, {'content': 'check out this document cat cat it is greater'},{'content': 'check out this document cat dog it is greater agiain this should work'}, {'content': 'check asd ffd ou asds asdt this document asdasd it is greater'}])
    df = pd.read_csv('../data/ufo_reports.csv')
    df['content'] = df['contents']
    nlp = NLP_vect(df.iloc[:1000])
    nlp.count_vect_func(stop_words= text.ENGLISH_STOP_WORDS.union('\n','\t'))
    nlp.tfidf_mat_func()
    test, train = nlp.split_func(nlp.count_vect)
    model = Model_UFO(nlp)
    model.PCA_func()

    # model.SVD_model()
    # model.scree_SVD()
    # model.cluster_SVD(model.SVD_mat)
