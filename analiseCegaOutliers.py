import numpy as np
import itertools
import random

#alteracao

def time_lagged_cov(X, num_lags):

    N = X.shape[0]
    L = X.shape[1] - num_lags
    R = np.empty([num_lags,N,N])

    center = lambda x: x-x.mean(1)[:,None]

    X0 = center(X[:, 0:(0+L)])
    
    for k in range(num_lags):
        Xk = center(X[:,k:(k+L)])
        R[k] = (1.0/L)*(X0.dot(Xk.T))
        R[k] = 0.5*(R[k] + R[k].T)

    return R


def prewhiten(X):

    # subtract mean
    Xw = X - X.mean(1)[:, None]

    # Compute SVD
    U,s,V = np.linalg.svd(Xw, full_matrices=False)
    Sinv = np.linalg.pinv(np.diag(s))

    # Find principal components
    Q  = np.dot(Sinv, U.T)
    Xw = Q.dot(X)

    return Q, Xw


def submat_mul(X, i, j, R, multype='post'):

    if multype.lower() == 'post':
        idx_i = (..., i)
        idx_j = (..., j)
    elif multype.lower() == 'pre':
        idx_i = (..., i, slice(None))
        idx_j = (..., j, slice(None))

     # In place multiplication X*R
    col_i = X[idx_i]*1.0
    col_j = X[idx_j]*1.0
    X[idx_i] = R[0,0]*col_i + R[0,1]*col_j
    X[idx_j] = R[1,0]*col_i + R[1,1]*col_j

    return X

def max_eigvec(A):
    # Find eigenvector associated with largest eigenvalue
    [eigvals,v] = np.linalg.eigh(A)
    v = v[:,-1]
    return v

def generate_index_pairs(idx_range, random_order=True):
    ij_pairs = itertools.combinations(idx_range, 2)
    ij_pairs = list(ij_pairs)
    if random_order:
        random.shuffle(ij_pairs)
    
    return ij_pairs

def off(X):
    off_ = 0
    for x in X:
        off_ += (np.linalg.norm(x,ord='fro')**2 - np.linalg.norm(np.diag(x))**2)

    return off_/np.max(X)

def jd(X, eps=1.0e-6, random_order = True):

    """jointly diagonalize several matrices.

    Performs jacobi-like procedure to approximately diagonalize
    a set of matrices X

    Parameters
    ----------
    X : np.ndarray
        Has dimensions [num_matrices, num_rows, num_cols]
    eps : scalar, optional
        Stopping criterion based on eps tolerance, should be between 0 and infinity
    random_order : bool, optional
        If True, pivots will cycle randomly for givens rotations.
        May affect convergence rate but not the final soltuion

    Returns
    -------
    V : np.ndarray
        2D array containing diagonalizing transformation
        i.e. (V.T).dot( X ).dot( V ) will be approximately diagonal
    
    """

    X = np.atleast_3d(X)
    V = np.eye(X.shape[1])

    keep_going = True
    counter = 0
    off_val = []
    while keep_going:
        print('{}: {}'.format(counter, off(X)))
        counter += 1
        keep_going = False

        ij_pairs = generate_index_pairs(range(X.shape[1]), 
                                        random_order = random_order)
        
        for (i,j) in ij_pairs:
            # Extract submatrix
            idx = (slice(None), ) + np.ix_([i,j],[i,j])
            A = X[idx]*1.0

            # Find givens rotation matrix
            R = find_givens_rotation(A)

            if abs(R[0,1]) > eps: # sin_theta = R[0,1]
                keep_going = True 
                # Update X and V matrices

                # X' = R^T X R
                X = submat_mul(X, i, j, R, multype='post')
                X = submat_mul(X, i, j, R, multype='pre')

                # V = V R
                V = submat_mul(V, i, j, R, multype='post')

    return V

def find_givens_rotation(A):

    """
    Belouchrani, A., et al. “A Blind Source Separation Technique Using 
    Second-Order Statistics.” IEEE Transactions on Signal Processing: 
    A Publication of the IEEE Signal Processing Society, vol. 45, 
    no. 2, Feb. 1997, pp. 434–44, doi:10.1109/78.554307.

    See Appendix A
    """
    G   = np.array( [ A[:,0,0] - A[:,1,1], A[:,0,1] + A[:,1,0] ] )
    G   = np.atleast_2d(G).T
    GHG = np.dot(G.T, G)
    v = max_eigvec(GHG)
    
    v = np.sign(v[0])*v
    cos_theta = np.sqrt(0.5 + 0.5*v[0])
    sin_theta = -0.5*v[1]/(cos_theta)

    R = np.array([[ cos_theta,  -sin_theta],
                    [ sin_theta, cos_theta]])

    return R

def sobi(X, num_lags=None, eps=1.0e-6, random_order = True):

    """blind source separation technique using SOBI algorithm

    The "second-order blind source idenitification" algorithm is 
    a blind-source separation technique that works by jointly diagonalizing
    a set of time-lagged covariance matrices. 

    Parameters
    ----------
    X : np.ndarray
        Has dimensions [num_signals, num_samples]
    num_lags : int
        Number of time-lags to use in forming covariance matrices
    eps : scalar, optional
        Stopping criterion based on eps tolerance, should be between 0 and infinity
    random_order : bool, optional
        If True, pivots will cycle randomly for givens rotations.
        May affect convergence rate but not the final soltuion

    Returns
    -------
    S : np.ndarray
        2D array containing estimated source signals
    
    A : np.ndarray
        2D array containing mixing matrix
        i.e. A.dot(S) = X
    
    W : np.ndarray
        2D array containing unmixing matrix
        i.e. W.dot(X) = S
    
    """

    if num_lags is None:
        num_lags = np.minimum(1000, int(X.shape[1]/2))

    Q, Xw = prewhiten(X)

    R = time_lagged_cov(Xw, num_lags)

    V = jd(R*1.0, eps=eps)

    W = (V.T).dot(Q)
    A = np.linalg.pinv(W)
    S = W.dot(X)

    return S, A, W


    
    



###########################################################################################################################


import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis


dados = pd.read_excel('C:/Users/ludimila/Desktop/TrabalhoPrático/ParaAlunosTrabCEGO/Outlier/X_outlier_100.xlsx')
mistura = pd.read_excel('C:/Users/ludimila/Desktop/TrabalhoPrático/AnaliseNCEGA/matrix_misturaA2.xlsx')

# Passo 1: Carregar os dados das duas fontes

X1 = np.array(dados['x1_2'])  # Dados da fonte 1
X2 = np.array(dados['x2_2'])  # Dados da fonte 2
#X3 = np.array(dados['s3_100k'])  # Dados da fonte 3
A = np.array(mistura)
tam = 100

plt.figure(1)
plt.suptitle('Fontes e histogramas')
plt.subplot(3,2,1)
plt.plot(X1, label = 'S1')
plt.xlim(0,tam)
plt.legend()

plt.subplot(3,2,2)
#plt.xlim(-1,1)
plt.hist(X1[1:tam], label = 'S1', bins = 40)
plt.legend()

plt.subplot(3,2,3)
plt.plot(X2, label = 'S2')
plt.xlim(0,tam)
plt.legend()

plt.subplot(3,2,4)
plt.hist(X2[1:tam], label = 'S2', bins = 40)
#plt.xlim(-1,1)
plt.legend()
'''
plt.subplot(3,2,5)
plt.plot(X3, label = 'S3')
plt.xlim(0,tam)
plt.legend()

plt.subplot(3,2,6)
plt.xlim(-4,4)
plt.hist(X3[1:tam], label = 'S3', bins = 40)
plt.legend()'''
plt.tight_layout()
plt.show()

print('\n')
print('correlação inicial S1,S2 = \n',np.corrcoef(X1,X2))
#print('correlação S1,S3 = \n',np.corrcoef(X1,X3))
#print('correlação S2,S3 = \n',np.corrcoef(X2,X3))
print('covariância inicial S1,S2 = \n',np.cov(X1,X2))
#print('covariância S1,S3 = \n',np.cov(X1,X3))
#print('covariância S2,S3 = \n',np.cov(X2,X3))
print('Kurtosis inicial X1: ', kurtosis(X1))
print('Kurtosis inicial X2: ', kurtosis(X2))

'''
# Passo 2: Definir a matriz de mistura
A = np.array([[0.2, 0.4],  # Relação de mistura para os sinais observados X1 e X2
              [0.4, 0.3]])'''

# Passo 3: Concatenar os arrays X1 e X2 para formar a matriz X
X = np.vstack((X1, X2))

# Passo 4: Chamar a função sobi
S, A_estimado, W_estimado = sobi(X)

print('s = ',S)
print('shape s = ', S.shape)
print('a estimado',A_estimado)
print('w estimado',W_estimado)


# S contém as estimativas dos sinais fonte
# A_estimado contém a matriz de mistura estimada
# W_estimado contém a matriz de separação estimada


plt.figure(2)
plt.suptitle('Fontes originais e estimadas')
plt.subplot(3,2,1)
plt.plot(X1, label = 'S1')
plt.xlim(0,tam)
plt.legend()

plt.subplot(3,2,2)
plt.plot(S[0,:], label = 'S estimada')
plt.xlim(0,tam)
#plt.legend()

plt.subplot(3,2,3)
plt.plot(X2, label = 'S2')
plt.xlim(0,tam)
plt.legend()

plt.subplot(3,2,4)
plt.plot(S[1,:], label = 'S estimada')
plt.xlim(0,tam)
#plt.legend()

'''plt.subplot(3,2,5)
plt.plot(X3, label = 'S3')
plt.xlabel('t')
plt.xlim(0,tam)
plt.legend()

plt.subplot(3,2,6)
plt.plot(S[2,:], label = 'S estimada')
plt.xlabel('t')
plt.xlim(0,tam)'''
#plt.legend()
plt.tight_layout()
plt.show()





plt.figure(3)
plt.suptitle('Fontes estimadas e histogramas')
plt.subplot(3,2,1)
plt.plot(S[0,:], label = 'S estimada')
plt.xlim(0,tam)
plt.legend()

plt.subplot(3,2,2)
plt.hist(S[0,1:tam], label = 'S estimada', bins = 40)

#plt.legend()

plt.subplot(3,2,3)
plt.plot(S[1,:], label = 'S estimada')
plt.xlim(0,tam)
plt.legend()

plt.subplot(3,2,4)
plt.hist(S[1,1:tam], label = 'S estimada', bins = 40)
#plt.xlim(0,tam)
#plt.legend()
'''
plt.subplot(3,2,5)
plt.plot(S[2,:], label = 'S estimada')
plt.xlabel('t')
plt.xlim(0,tam)
plt.legend()

plt.subplot(3,2,6)
plt.hist(S[2,1:tam], label = 'S estimada', bins = 40)
plt.xlabel('t')
#plt.legend()'''
plt.tight_layout()
plt.show()


print('\n')
print('correlação final S1,S2 = \n',np.corrcoef(S[:,0],S[:,1]))
#print('correlação S1,S3 = \n',np.corrcoef(S[:,0],S[:,2]))
#print('correlação S2,S3 = \n',np.corrcoef(S[:,1],S[:,2]))
print('covariância final S1,S2 = \n',np.cov(S[:,0],S[:,1]))
#print('covariância S1,S3 = \n',np.cov(S[:,0],S[:,2]))
#print('covariância S2,S3 = \n',np.cov(S[:,1],S[:,2]))
print('Kurtosis final X1: ', kurtosis(S[:,0]))
print('Kurtosis final X2: ', kurtosis(S[:,1]))