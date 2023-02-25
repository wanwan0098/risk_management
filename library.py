import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize


# 1. covariance related
def corr(cov):       
    invStd = np.diag(1.0/np.sqrt(np.diag(cov)))
    return invStd @ cov @ invStd

def var(cov):
    return np.diag(cov)

def cov(corr, var):
    std = np.diag(np.sqrt(var))
    return std@corr@std

def exp_weighted_cov(returns, lambd):
    t = len(returns)
    weight = np.zeros(t)
    for i in range(1,t+1):
        weight[t-i] = (1-lambd)*lambd**(i-1)
    weight = weight/sum(weight)
    returns = returns - returns.mean()
    return returns.T @ (np.diag(weight) @ returns)


# 2. cholesky factorization, non-PSD fixes, non-PSD generation, check if PSD
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
    - A: cov matrix
    - w: a diagonal matrix, set to identity matrix if unweighted
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


# 3. simulation methods
def direct_sim(cov,nsim=25000):
    np.random.seed(9)
    return chol_psd(cov)@np.random.normal(size=(len(cov),nsim))

# pca
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
    # np.random.seed(9)
    z = np.random.normal(size = (len(vals),nsim))
    return B @ z

def sim_CBM(num,p0,sigma=0.2,mu=0):
    np.random.seed(8)
    r = np.random.normal(mu,sigma,num)
    p = np.zeros(num)
    for i in range(num):
        p[i] = p0 + r[i]
    return np.mean(p), np.std(p)

def sim_ARS(num,p0,sigma=0.2,mu=0):
    np.random.seed(8)
    r = np.random.normal(mu,sigma,num)
    p = np.zeros(num)
    for i in range(num):
        p[i] = p0*(1+r[i])
    return np.mean(p), np.std(p)

def sim_GBM(num,p0,sigma=0.2,mu=0):
    np.random.seed(8)
    r = np.random.normal(mu,sigma,num)
    p = np.zeros(num)
    for i in range(num):
        p[i] = p0*np.exp(r[i])
    return np.mean(p), np.std(p)


# 4. VaR
# A. Single asset VaR using different distributions
# B. Portfolio VaR through Delta Normal, Monte Carlo, and Historical
def var(returns, alpha=0.05):
    # returns = returns-returns.mean()
    return -np.percentile(returns,alpha*100)

def var_normal(returns, num_sim=10000, alpha=0.05):
    # np.random.seed(8)
    # returns = returns-returns.mean()
    sim_returns = np.random.normal(np.mean(returns),np.std(returns),size=num_sim)
    var_normal = -np.percentile(sim_returns,alpha*100)
    return [sim_returns, var_normal]

def var_exp_weighted(returns, num_sim=10000, alpha=0.05):
    returns = returns-returns.mean()
    sigma = np.sqrt(exp_weighted_cov(returns, 0.94))
    sim_returns = np.random.normal(np.mean(returns),sigma,size=num_sim)
    var_exp_weighted = -np.percentile(sim_returns,alpha*100)
    return [sim_returns, var_exp_weighted]

def t_ll(params, returns):
    df = params[0]
    loc = params[1]
    scale = params[2]
    ll = stats.t.logpdf(returns, df, loc, scale).sum()
    return -ll

def var_t_MLE(returns,num_sim=10000, alpha=0.05):
    # returns = returns-returns.mean()
    constraints=({"type":"ineq", "fun":lambda x: x[0]-1}, {"type":"ineq", "fun":lambda x: x[2]})
    returns = minimize(t_ll, [10, np.mean(returns), np.std(returns)], args=returns, constraints=constraints)
    df, loc, scale = returns.x[0],returns.x[1],returns.x[2]
    dist = stats.t(df=df,loc=loc,scale=scale)
    sim_returns = dist.rvs(size=num_sim)
    var_t_mle = -np.percentile(sim_returns, alpha*100)
    return [dist, sim_returns, var_t_mle]

def var_ar1(returns,num_sim=10000, alpha=0.05):
    ar1 = sm.tsa.ARIMA(returns,order=(1,0,0)).fit()
    alpha_ar1 = ar1.params[0]
    resid = ar1.resid
    sigma = np.std(resid)
    sim_returns = []
    np.random.seed(8)
    for i in range(num_sim):
        sim_returns.append(alpha_ar1*(returns.iloc[-1])+np.random.normal(0,sigma))
    var_ar1 = np.percentile(sim_returns, alpha*100)
    return [sim_returns, var_ar1]

def var_hist(returns, alpha=0.05):
    var_hist = np.percentile(returns, alpha*100)
    return [returns, var_hist]

def parametric_VAR(portfolio, prices, alpha=0.05, lambd = 0.94):
    '''
    portfolio: a dataframe with columns 'Stock' and 'Holding'
    prices: a dataframe with columns as stocks and rows as time points
    '''
    current_prices, holdings, portfolio_value = calc_portfolio_value(portfolio, prices)
    delta = np.identity(portfolio.shape[0])
    gradient = current_prices/portfolio_value*(delta@holdings)
    prices = prices.loc[:,portfolio['Stock']]
    returns = return_calculate(prices).reset_index()
    returns.drop('index',axis=1,inplace=True)
    cov = exp_weighted_cov(returns, lambd)
    VaR = -portfolio_value * stats.norm.ppf(alpha) * np.sqrt(gradient.T @ cov @ gradient)
    return VaR

def monte_carlo_VAR(portfolio, prices, nsim=20000, lambd=0.94, alpha=0.05):
    current_prices, holdings, portfolio_value = calc_portfolio_value(portfolio, prices)
    
    prices = prices.loc[:,portfolio['Stock']]
    returns = return_calculate(prices).reset_index()
    returns.drop('index',axis=1,inplace=True)
    cov = exp_weighted_cov(returns, lambd)
    
    np.random.seed(66)
    sim_returns = multivariate_normal_sim(cov, nsim)
    sim_pv = (current_prices*holdings).T@sim_returns
    VaR = -np.percentile(sim_pv, alpha*100)
    return [sim_pv, VaR]

def historical_VAR(portfolio, prices, nsim=2000, lambd=0.94, alpha=0.05):
    current_prices, holdings, portfolio_value = calc_portfolio_value(portfolio, prices)
    
    prices = prices.loc[:,portfolio['Stock']]
    returns = return_calculate(prices).reset_index()
    returns.drop('index',axis=1,inplace=True)
    
    np.random.seed(99)
    sim_returns = returns.sample(nsim, replace=True)
    sim_pv = (current_prices*holdings).T@(sim_returns.T)
    VaR = -np.percentile(sim_pv, alpha*100)
    return [sim_pv,VaR]


# 5. ES
def es(returns, alpha=0.05):
    returns.sort()
    index = round(alpha*len(returns))
    return -np.mean(returns[:index+1])


# 6. portfolio related
def return_calculate(prices, method='DISCRETE'):
    price = prices.pct_change().dropna()
    if method.upper() == "DISCRETE":
        return price
    elif method.upper() == "LOG":
        return np.log(price)

def calc_portfolio_value(portfolio, prices):
    dict_current_prices = prices.iloc[-1,:].to_dict()
    portfolio['current_prices'] = portfolio['Stock'].map(dict_current_prices)
    current_prices = portfolio['current_prices'].values
    holdings = portfolio['Holding'].values
    portfolio_value = np.dot(holdings, current_prices)
    return current_prices, holdings, portfolio_value
