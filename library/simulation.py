import pandas as pd
import numpy as np
import psd
from scipy import stats
import statistics as stat
import VaR


# simulate directly from a covariance matrix 
def direct_sim(cov,nsim=25000):
    if not psd.is_psd(cov):
        cov = psd.near_psd(cov)
    return psd.chol_psd(cov)@np.random.normal(size=(len(cov),nsim))

# simulate using pca to specify % variance explained
def multivariate_normal_sim(cov, nsim, var_explained=1-1e-9):
    vals, vecs = np.linalg.eigh(cov)
    tot = sum(vals)
    for i in range(len(vals)):
        i = len(vals)-1-i
        if sum(vals[i:])/tot > var_explained:
            vals = vals[i:]
            vecs = vecs[:,i:]
            break
        if vals[i]<0:
            vals = vals[i+1:]
            vecs = vecs[:,i+1:]
            break
    B = vecs @ np.diag(np.sqrt(vals))
    z = np.random.normal(size = (len(vals),nsim))
    return B @ z

# breaking the normality assumption
# used to combine variables with different assumed distributions
def gaussian_copula(returns, fitted_models, num_sim):
    '''
    returns: DataFrame of data to be fitted
    fitted_models: np.array of types of models: Normal or T
    '''

    n=returns.shape[1]
    assets = returns.columns.tolist()
    assets_cdf = pd.DataFrame()
    params = []

    for i, col in enumerate(assets):
        # returns[col] -= returns[col].mean()
        if fitted_models[i]=='T':
            t_res = t_sim(returns[col])
            dof, mean, scale = t_res[1],t_res[2],t_res[3]
            fitted = 'T'
            params.append([dof, mean, scale])
        elif fitted_models[i]=='Normal':
            x_mean = returns[col].mean()
            x_std = np.sqrt(stat.variance(returns[col]))
            fitted = 'Normal'
            params.append([x_mean, x_std])
        
        if fitted=='T':
            assets_cdf[col] = stats.t.cdf(returns[col], df=dof, loc=mean, scale=scale)
        elif fitted=='Normal':
            assets_cdf[col] = stats.norm.cdf(returns[col], loc=x_mean, scale=x_std)
        #U
    
    # Because Spearman uses Rank correlations and the quantile function is monotonic, we can skip calculating 
    # Z values (Z = stats.norm.ppf(U)), and calculate Spearman correlations from the U matrix directly.

    corr_spearman = stats.spearmanr(assets_cdf)[0]

    # copula = stats.multivariate_normal(mean=np.zeros(len(ret.columns)), cov=corr_spearman)
    # spearman = pd.DataFrame(copula.rvs(size=num_sim))
    spearman = pd.DataFrame(multivariate_normal_sim(corr_spearman, 1000).T)

    sim_returns = pd.DataFrame()
    for i in range(n):
        simu = stats.norm.cdf(spearman.iloc[:, i])
        if fitted_models[i]=='T':
            sim_returns[returns.columns[i]] = stats.t.ppf(simu, df=params[i][0], loc=params[i][1], scale=params[i][2])
        elif fitted_models[i]=='Normal':
            sim_returns[returns.columns[i]] = stats.norm.ppf(simu, loc=params[i][0], scale=params[i][1])

    return sim_returns

# now useless
def copula_t_sim(returns, num_sim):
    '''
    returns: DataFrame without Date
    '''

    n=returns.shape[1]
    stock_cdf = pd.DataFrame()
    t_params = []

    for col in returns.columns:
        returns[col] -= returns[col].mean()
        t_res = t_sim(returns[col])
        dof, mean, scale = t_res[1],t_res[2],t_res[3]
        t_params.append([dof, mean, scale])
        stock_cdf[col] = stats.t.cdf(returns[col], df=dof, loc=mean, scale=scale) #U
    
    # Because Spearman uses Rank correlations and the quantile function is monotonic, we can skip calculating 
    # Z values (Z = stats.norm.ppf(U)), and calculate Spearman correlations from the U matrix directly.

    corr_spearman = stats.spearmanr(stock_cdf)[0]
    
    # copula = stats.multivariate_normal(mean=np.zeros(len(ret.columns)), cov=corr_spearman)
    # spearman = pd.DataFrame(copula.rvs(size=num_sim))
    spearman = pd.DataFrame(multivariate_normal_sim(corr_spearman, num_sim, var_explained=1-1e-9).T)

    sim_returns = pd.DataFrame()
    for i in range(n):
        simu = stats.norm.cdf(spearman.iloc[:, i])
        sim_returns[returns.columns[i]] = stats.t.ppf(simu, df=t_params[i][0], loc=t_params[i][1], scale=t_params[i][2])

    return sim_returns

# simulate from a t-dist by MLE fit (single asset)
def t_sim(returns):
    res = VaR.var_t_MLE(returns)
    sim_returns = res[1]
    dof, loc, scale = res[3], res[4], res[5]
    return sim_returns, dof, loc, scale


def sim_CBM(num,p0,sigma=0.2,mu=0):
    r = np.random.normal(mu,sigma,num)
    p = np.zeros(num)
    for i in range(num):
        p[i] = p0 + r[i]
    return np.mean(p), np.std(p)

def sim_ARS(num,p0,sigma=0.2,mu=0):
    r = np.random.normal(mu,sigma,num)
    p = np.zeros(num)
    for i in range(num):
        p[i] = p0*(1+r[i])
    return np.mean(p), np.std(p)

def sim_GBM(num,p0,sigma=0.2,mu=0):
    r = np.random.normal(mu,sigma,num)
    p = np.zeros(num)
    for i in range(num):
        p[i] = p0*np.exp(r[i])
    return np.mean(p), np.std(p)
