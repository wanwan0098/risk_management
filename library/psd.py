import numpy as np

# cholesky factorization, non-PSD fixes, non-PSD generation, check if PSD
def chol_psd(a):
    '''
    Return the cholesky root given a symmetric, PSD matrix
    '''
    a = np.array(a)
    root = np.zeros_like(a)
    for j in range(len(a)):
        for i in range(j,len(a)):
            if i==j:
                root[i,j] = a[i,j]-np.dot(root[j,:j],root[j,:j])
                if abs(root[i,j]) <= 1e-9:
                    root[i,j] = 0
                else:
                    root[i,j] = np.sqrt(root[i,j])
            else:
                if root[j,j] == 0:
                    root[i,j] = 0
                else: root[i,j] = (a[i,j]-np.dot(root[i,:j],root[j,:j]))/root[j,j]
    return root

# Rebonato and Jackel
def near_psd(a, epsilon=0.0):
    '''
    Return a near PSD covariance matrix given a non-PSD correlation or covariance matrix
    '''
    cov = False
    for i in np.diag(a):
        if abs(i-1)>=1e-9:
            cov = True
    if cov:
        invStd = np.diag(1.0/np.sqrt(np.diag(a)))
        a = invStd @ a @ invStd
    vals, vecs = np.linalg.eigh(a)
    vals = [val if val>=0.0 else epsilon for val in vals]
    T = 1.0/((np.square(vecs) @ vals))
    B = np.diag(np.sqrt(T)) @ vecs @ np.diag(np.sqrt(vals))
    res = B@B.T
    if cov:
        std = np.diag(1.0/np.diag(invStd))
        res = std @ res @ std
    return res

def Ps(A, w):
    A = np.sqrt(w)@A@np.sqrt(w)
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, a_min=0, a_max=None)
    return np.sqrt(w)@(vecs@np.diag(vals)@vecs.T)@np.sqrt(w)

def Pu(A): # paper p336, 3.2
    res = A.copy()
    for i in range(len(A)):
        res[i][i]=1
    return res

def frobenius(a, pu_ps_a): # class notes p9
    d = pu_ps_a - a
    s = 0
    for i in range(len(d)):
        for j in range(len(d)):
            s+=d[i][j]**2
    return s

def higham_psd(A, w, max_iter=1000,tol=1e-9, print_dif=False):
    '''
    Return a near PSD corr matrix given a non-PSD corr matrix
    
    Parameters:
    - A: cov matrix (np.array)
    - w: a diagonal matrix, set to identity matrix if unweighted: np.identity(len(A))
    - max_iter: cap on the iterations
    - tol: the norm difference at which the alternative projections are stopped
    - print_dif: print norm differences for each iteration
    '''
    prev_norm = float("inf")
    y = A.copy()
    delta_s = np.zeros(A.shape)
    for i in range(max_iter):
        r = y-delta_s
        x = Ps(r,w)
        delta_s = x-r
        y = Pu(x)
        norm = frobenius(A,y)
        if print_dif:
            print(abs(norm-prev_norm))
        if abs(norm-prev_norm)<tol:
            break
        else:
            prev_norm = norm
    return y

def non_psd(n=500):
    sigma = np.full((n,n),0.9)
    for i in range(n):
        sigma[i, i] = 1.0
    sigma[0, 1] = 0.7357
    sigma[1, 0] = 0.7357
    return sigma

def is_psd(mat):
    return np.all(np.linalg.eigvals(mat) > -1e-8)
