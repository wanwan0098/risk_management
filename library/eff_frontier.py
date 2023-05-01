import pandas as pd
import numpy as np
from scipy.optimize import minimize


# maximum sharpe ratio portfolio
# caveat: weights could swap when num_assets is 2 and short sale is allowed with a lower bound
#         check them against avg_returns or use brute force (last function)
def super_efficient_portfolio(avg_returns, rf, cov_matrix, shortAllowed=False):
    """
    params:
    avg_returns: asset average return, size n x 1
    cov_matrix: portfolio covariance, size n x n
    rf: risk-free rate
    shortValid: whether short sale is allowed
    """
    
    num_assets = avg_returns.shape[1] if len(avg_returns.shape) > 1 else avg_returns.shape[0]
    
    def neg_sharpe_ratio(weights):
        port_return = avg_returns.dot(weights)
        port_std_dev = np.sqrt(weights.T.dot(cov_matrix).dot(weights))
        sharpe = (port_return - rf) / port_std_dev
        return -sharpe
    
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
    bounds = [(0, 1) for _ in range(num_assets)]
    
    init_weights = np.ones(num_assets) / num_assets
    if shortAllowed == False:
        opt_result = minimize(neg_sharpe_ratio, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    else:
        opt_result = minimize(neg_sharpe_ratio, init_weights, method='SLSQP', constraints=constraints)
    
    opt_weights = opt_result.x
    opt_port_return = avg_returns.dot(opt_weights)
    opt_port_std_dev = np.sqrt(opt_weights.T.dot(cov_matrix).dot(opt_weights))
    opt_sharpe_ratio = (opt_port_return - rf) / opt_port_std_dev
    try:
        opt_weights = pd.DataFrame(opt_weights, index=avg_returns.index, columns=['Weight'])
    except: return opt_weights, opt_sharpe_ratio
    return opt_weights, opt_sharpe_ratio


def efficient_frontier(expected_returns, cov_matrix, target_return):
    def portfolio_volatility(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    n_assets = len(expected_returns)

    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}, 
                   {'type': 'eq', 'fun': lambda weights: np.sum(weights * expected_returns) - target_return}]
    bounds = [(0, 1) for _ in range(n_assets)]

    initial_guess = np.array([1/n_assets for _ in range(n_assets)])
    efficient = minimize(portfolio_volatility, initial_guess, 
                         args=(expected_returns, cov_matrix), 
                         method='SLSQP', 
                         constraints=constraints, 
                         bounds=bounds)

    # volatility, weights
    return efficient.fun, efficient.x

def calculate_efficient_frontier(expected_returns, cov_matrix, target_returns):
    frontiers = []
    for target_return in target_returns:
        frontier, _ = efficient_frontier(expected_returns, cov_matrix, target_return)
        frontiers.append(frontier)
        eff_frontiers = pd.DataFrame({'vol': frontiers, 'returns': target_returns})
    return eff_frontiers


# maximum risk/return using ES 
def efficient_es(returns, weights, alpha=0.05):
    returns = returns.dot(weights)
    returns.sort_values(inplace=True)
    index = round(alpha*len(returns))
    return -np.mean(returns[:index+1])

def super_efficient_portfolio_es(returns, avg_returns, rf, shortAllowed=False):
    """
    params:
    returns: t x n
    avg_returns: asset average return, size n x 1
    """
    num_assets = avg_returns.shape[1] if len(avg_returns.shape) > 1 else avg_returns.shape[0]
    
    def neg_sharpe_ratio_es(weights):
        port_return = avg_returns.dot(weights)
        es = efficient_es(returns, weights)
        sharpe_es = (port_return - rf) / es
        return -sharpe_es
    
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
    bounds = [(0, 1) for _ in range(num_assets)]
    bounds2 = [(-1, None) for _ in range(num_assets)] # self defined (leverage of 1 here)
    
    init_weights = np.ones(num_assets) / num_assets
    if shortAllowed == False:
        opt_result = minimize(neg_sharpe_ratio_es, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    else:
        opt_result = minimize(neg_sharpe_ratio_es, init_weights, method='SLSQP', bounds=bounds2, constraints=constraints)
    
    opt_weights = opt_result.x
    return opt_weights


def opt_weights_two_assets(assets, avg_returns, cov, rf, bound=(0,1)):
    def sr(w):
        m = np.dot(w, avg_returns) - rf
        s = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        return m/s

    rng = np.arange(bound[0],bound[1]+0.00001,0.0001)
    vals = np.zeros((len(rng), 3))

    for i, a in enumerate(rng):
        s = sr(np.array([a, 1 - a]))
        vals[i,:] = [a, 1 - a, s]

    max_idx = np.argmax(vals[:, 2])
    result = vals[max_idx, [0, 1]]
    opt_weights = pd.DataFrame({"Weight":result}, index=assets)
    return opt_weights, sr(result)
