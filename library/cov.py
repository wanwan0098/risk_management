import numpy as np
import numpy.ma as ma

def corr(cov):       
    invStd = np.diag(1.0/np.sqrt(np.diag(cov)))
    return invStd @ cov @ invStd

def var(cov):
    return np.diag(cov)

def cov(corr, var):
    std = np.diag(np.sqrt(var))
    return std@corr@std

def exp_weighted_cov(returns, lambd):
    '''
    returns: DF w/o Date
    lambd: lower lambda = current values weighed more heavily
    '''
    t = len(returns)
    weight = np.zeros(t)
    for i in range(1,t+1):
        weight[t-i] = (1-lambd)*lambd**(i-1)
    weight = weight/sum(weight)
    returns = returns - returns.mean()
    return returns.T @ (np.diag(weight) @ returns)

def exp_weighted_matrix(returns, lambd):
    # returns: DF
    returns = returns.values
    n_timesteps = returns.shape[0]
    weights = np.zeros(n_timesteps)
    for t in range(n_timesteps):
        weights[n_timesteps-1-t]  = (1-lambd)*lambd**t
    weights_matrix = np.diag(weights/sum(weights))
    return weights_matrix

def missing_cov(x, skipMiss=True, fun=np.corrcoef):
    '''
    skipMiss=True: contains only rows that contain no missing values in any column
    skipMiss=False: computes the covariance between all pairs of variables, using only the
    rows that have non-missing values for both variables
    '''
    n, m = x.shape
    nMiss = np.sum(np.isnan(x), axis=0)

    # nothing missing, just calculate it.
    if np.sum(nMiss) == 0:
        return fun(x)

    idxMissing = [set(np.where(np.isnan(x[:, i]))[0]) for i in range(m)]

    if skipMiss:
        # Skipping Missing, get all the rows which have values and calculate the covariance
        rows = set(range(n))
        for c in range(m):
            for rm in idxMissing[c]:
                if rm in rows:
                    rows.remove(rm)
        rows = sorted(list(rows))
        return fun(x[rows,:].T)

    else:
        # Pairwise, for each cell, calculate the covariance.
        out = np.empty((m, m))
        for i in range(m):
            for j in range(i+1):
                rows = set(range(n))
                for c in (i,j):
                    for rm in idxMissing[c]:
                        if rm in rows:
                            rows.remove(rm)
                rows = sorted(list(rows))
                out[i,j] = fun(x[rows,:][:,[i,j]].T)[0,1]
                if i != j:
                    out[j,i] = out[i,j]
        return out
