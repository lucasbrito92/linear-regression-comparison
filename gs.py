#ref: http://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
import numpy as np
 
def proj(u, v):
    # notice: this algrithm assume denominator isn't zero
    return u * np.dot(v,u) / np.dot(u,u)  
 
def GS(V):
    V = 1.0 * V     # to float
    U = np.copy(V)
    for i in range(1, V.shape[1]):
        for j in range(i):
            U[:,i] -= proj(U[:,j], V[:,i])
    # normalize column
    den=(U**2).sum(axis=0) **0.5
    E = U/den
    # assert np.allclose(E.T, np.linalg.inv(E))
    return E
    
#if __name__ == '__main__':
#    V = np.array([[1.0, 1, 1], [1, 0, 2], [1, 0, 0]]).T