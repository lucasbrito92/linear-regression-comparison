# useful links to this program
#
# https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f
# https://archive.ics.uci.edu/ml/datasets/abalone

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

column_names = ["sex", "length", "diameter", "height", "whole weight", 
                "shucked weight", "viscera weight", "shell weight", "rings"]

data = pd.read_csv("abalone.data", names=column_names)

print("Number of samples: %d" % len(data))
print(data.head())

y = data.rings.values

del data["sex"] # removendo a primeira característica
del data["rings"] # remove rings from data, so we can convert all the dataframe to a numpy 2D array

X = data.values.astype(np.float)

print(data.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

#linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
lmreg = linear_model.LinearRegression( fit_intercept = False )
#lmreg = linear_model.LinearRegression( fit_intercept = True )
#lmreg = linear_model.LinearRegression( fit_intercept = True, normalize = True )

lmreg.fit(X_train, y_train)

print('Coeficientes da regressão: ', lmreg.coef_)

predicted_y_test = lmreg.predict(X_test)
predicted_y_train = lmreg.predict(X_train)

scatter_y(y_train, predicted_y_train)
plt.title("Conjunto de Treinamento")
scatter_y(y_test, predicted_y_test)
plt.title("Conjunto de Teste")

plt.show()