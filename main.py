# useful links to this program
#
# https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f
# https://archive.ics.uci.edu/ml/datasets/abalone
# http://www.blackarbs.com/blog/intro-to-expectation-maximization-k-means-gaussian-mixture-models-with-python-sklearn/3/20/2017

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pow
import operator

def gs(vectors):
    basis = []
    for v in vectors:
        w = v - np.sum( np.dot(v,b)*b  for b in basis )
        if (w > 1e-10).any():  
            basis.append(w/np.linalg.norm(w))
    return np.array(basis)

def qr_fit(A, b):

    # do QR factorization
    Q, R = np.linalg.qr(A)
    x_til = np.dot(np.dot(np.linalg.inv(R), Q.T), b)

    return x_til

def qr_predict(A_train, qr_coef):

    # mutiply A*x~ = b to predict rings in abalone dataset
    b_train = []

    for features in A_train:
        b_train.append(np.dot(features, qr_coef))

    return np.array(b_train)

def scatter_y(true_y, predicted_y):
    """
    Scatter-plot the predicted vs true number of rings

    Plots:
       * predicted vs true number of rings
       * perfect agreement line
       * +2/-2 number dotted lines

    Returns the root mean square of the error
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(true_y, predicted_y, '.k')

    ax.plot([0, 30], [0, 30], '--k')
    ax.plot([0, 30], [2, 32], ':k')
    ax.plot([2, 32], [0, 30], ':k')

    rms = (true_y - predicted_y).std()

    ax.text(25, 3,
            "Root Mean Square Error = %.2g" % rms,
            ha='right', va='bottom')

    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)

    ax.set_xlabel('Número real de anéis')
    ax.set_ylabel('Número predito de anéis')

    return rms

def cos_func(feature, output):

    cos = pow((np.dot(feature, output)), 2)/(np.linalg.norm(feature)*np.linalg.norm(output))
    return cos

def project_sub(v, u):

    for i in range(u.shape[0]):
        s += np.dot(v, u[i])*u[i]

    return s

def tst(A):

    rank = []
    for i in range(A.shape[1]):
        rank.append((cos_func(X.T[i], y), i))
    
    # sort by greater values
    rank.sort(key = operator.itemgetter(0), reverse=False)
    return rank[0]

if __name__ == "__main__":

    column_names = ["sex", "length", "diameter", "height", "whole weight",
                    "shucked weight", "viscera weight", "shell weight", "rings"]

    data = pd.read_csv("abalone.data", names=column_names)

    print("Number of samples: %d" % len(data))
    # print(data.head())

    y = data.rings.values

    del data["sex"]
    del data["rings"]

    X = data.values.astype(np.float)

    for i in range(X.shape[1]):
        selected = tst(X)
        #= project_sub(selected, )
    #print(rank.sort())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0)

    # linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
    lmreg = linear_model.LinearRegression(fit_intercept=False)
    # lmreg = linear_model.LinearRegression( fit_intercept = True )
    # lmreg = linear_model.LinearRegression( fit_intercept = True, normalize = True )

    lmreg.fit(X_train, y_train)
    #print('Sklearn Linear Regression: ', lmreg.coef_)

    predicted_y_test = lmreg.predict(X_test)
    predicted_y_train = lmreg.predict(X_train)

    rms_test = (y_test - predicted_y_test).std()
    rms_train = (y_train - predicted_y_train).std()

    print('Y_test_rms: ', rms_test)
    print('Y_train_rms: ', rms_train)

    # Least squares w/ Gram-schimidt

    qr_coef = qr_fit(X_train, y_train)
    #print('Least squares w/ Gram-schimidt : ', qr_coef)

    predicted_y_test_qr = qr_predict(X_test, qr_coef)
    predicted_y_train_qr = qr_predict(X_train, qr_coef)

    rms_test_qr = (y_test - predicted_y_test_qr).std()
    rms_train_qr = (y_train - predicted_y_train_qr).std()

    print('Y_test_rms: ', rms_test_qr)
    print('Y_train_rms: ', rms_train_qr)

    """ scatter_y(y_train, predicted_y_train)
    plt.title("Conjunto de Treinamento")

    scatter_y(y_test, predicted_y_test)
    plt.title("Conjunto de Teste")
 
    plt.show()
"""
