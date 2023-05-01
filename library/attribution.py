import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
import es


def update_weights(weights, returns):
    """
    Update the weights of each asset in a portfolio given the initial
    weight and returns. The initial weight and returns start at the
    same period.
    params:
        - weights: DataFrame, shape(, n); could be the optimal weights from the super efficient frontier
        - returns: DataFrame, shape(t, n)
    return:
        - weighted_rets: DataFrame, shape(t, n)
    """
    latest_weights = weights.values[0].copy()
    updated_weights = np.zeros(shape=(returns.shape[0], len(latest_weights)))
    for i in range(returns.shape[0]):
        updated_weights[i, :] = latest_weights
        latest_weights *= (1 + returns.iloc[i, :])
        latest_weights /= sum(latest_weights)
    updated_weights = pd.DataFrame(updated_weights ,index = returns.index ,columns = returns.columns)
    weighted_rets = updated_weights*returns.values
    return weighted_rets

def cal_carino_k(weighted_rets_t):
    R = (weighted_rets_t +1).prod(axis=0 ) -1
    K = np.log( 1 +R ) /R
    kt = np.log( 1 +weighted_rets_t ) /( K *weighted_rets_t)
    return kt

def return_attribution(weighted_rets, ex='post'):
    '''
    weighted_rets: t x n
    weighted_rets_t: t x 1
    cal_carino_k(weighted_rets_t): t x 1; dimension: (n,) which could be treated as (,n)
    '''
    weighted_rets_t = weighted_rets.sum(axis=1)
    res = cal_carino_k(weighted_rets_t)
    if ex=='post':
        return res @ weighted_rets
    elif ex=='ante':
        return res


# ex-post risk attribution
def risk_attribution(weighted_rets):
    weighted_rets_t = weighted_rets.sum(axis=1)
    port_sigma = weighted_rets_t.std()

    risk_attribution = {}
    for st in weighted_rets.columns:
        # ri = alpha + sum(beta * rt) + error
        model = sm.OLS(weighted_rets[st], sm.add_constant(weighted_rets_t))
        results = model.fit()
        risk_attribution[st] = results.params.values[1] * port_sigma
    return risk_attribution.values()


def cal_port_vol(weights, cov):
    w = np.array(weights).flatten()
    port_std = np.sqrt(w@cov@w)
    return port_std

# ex-ante risk attribution
def cal_component_std(weights, cov, equal=True, risk_budget=None):
    w = np.array(weights).flatten()
    port_std = cal_port_vol(weights, cov)
    csd = w * (cov@w)/port_std  # weight * gradient
    if equal==False:
        csd = csd/risk_budget
    return csd

# def norm_risk_attribution(weights, cov): ## risk budget
#     port_std = cal_port_vol(weights, cov)
#     csd = cal_component_std(weights, cov)
#     return csd/port_std

def cal_csd_sse(weights, cov, equal=True, risk_budget=None):
    csd = cal_component_std(weights, cov, equal, risk_budget)
    csd_ = csd - np.mean(csd)
    return sum(csd_**2)

def risk_budget_weights(cov, equal=True, risk_budget=None):
    '''
    equal: if equal is true, equal risk budgets / risk parity
    risk_budget: np.array of size num_assets if equal is false
    '''

    num_assets = cov.shape[0]
    initial_wts = np.array(num_assets * [1. / num_assets])
    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
    cov = np.array(cov)
    bnds = tuple((0, 1) for _ in range(num_assets))
    opt_wts = minimize(lambda w: 1e5*cal_csd_sse(w, cov, equal, risk_budget), initial_wts,bounds=bnds, constraints=cons)
    return opt_wts.x

def risk_budget_sharpe(opt_weights, avg_returns, rf, cov):
    # opt_wts.x
    opt_port_return = avg_returns.dot(opt_weights)
    opt_port_std_dev = np.sqrt(opt_weights.T.dot(cov).dot(opt_weights))
    opt_sharpe_ratio = (opt_port_return - rf) / opt_port_std_dev
    return opt_sharpe_ratio


# non-normal risk parity: ES as the measure
def cal_port_es(weights, rets):
    w = np.array(weights).flatten()
    return es.es(rets@w, df=False)

def cal_component_es(weights, rets):
    w = np.array(weights).flatten()
    e = 1e-6
    num_assets = len(weights)
    es = cal_port_es(w, rets)
    ces = np.zeros(num_assets)

    for i in range(num_assets):
        weight_i = w[i]
        w[i] += e
        ces[i] = weight_i * (cal_port_es(w, rets) - es) / e
        w[i] -= e
    return ces

def cal_ces_sse(weights, rets):
    ces = cal_component_es(weights, rets)
    ces_ = ces - np.mean(ces)
    return ces_@ces_.T

def risk_parity_es_weights(rets):
    # rets: DataFrame without Date
    num_assets = rets.shape[1]
    initial_wts = np.array(num_assets * [1. / num_assets])
    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
    rets = np.array(rets)
    bnds = tuple((0, 1) for _ in range(num_assets))
    opt_wts = minimize(lambda w: 1e5*cal_ces_sse(w, rets), initial_wts,bounds=bnds, constraints=cons)
    return opt_wts.x
