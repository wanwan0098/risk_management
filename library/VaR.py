import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize
import cov
import portfolio_calc
import simulation

# VaR
# A. Single asset VaR simulated using different distributions
# B. Portfolio VaR, in dollars, through Delta Normal, Monte Carlo, and Historical simulations
# C. Portfolio VaR calculation (assuming all assets ~ Normal)

def var(returns, alpha=0.05):
    return -np.percentile(returns,alpha*100)

def var_normal(returns, num_sim=10000, alpha=0.05):
    sim_returns = np.random.normal(np.mean(returns),np.std(returns),size=num_sim)
    var_normal = -np.percentile(sim_returns,alpha*100)
    return var_normal, sim_returns

def var_exp_weighted(returns, num_sim=10000, alpha=0.05):
    sigma = np.sqrt(cov.exp_weighted_cov(returns, 0.94))
    sim_returns = np.random.normal(np.mean(returns),sigma,size=num_sim)
    var_exp_weighted = -np.percentile(sim_returns,alpha*100)
    return var_exp_weighted, sim_returns

def var_t_MLE(returns,num_sim=10000, alpha=0.05):
    def t_ll(params, returns):
        dof, loc, scale = params[0], params[1], params[2]
        ll = stats.t.logpdf(returns, dof, loc, scale).sum()
        # the total log-likelihood of all the observed returns
        return -ll
    # the objective function to be minimized: fun(x,*args) -> float
    constraints=({"type":"ineq", "fun":lambda x: x[0]-1}, {"type":"ineq", "fun":lambda x: x[2]})
    # degrees of freedom > 1 and the scale parameter > 0
    res = minimize(t_ll, [10, np.mean(returns), np.std(returns)], args=returns, constraints=constraints)
    # fun, initial guesses, args
    dof, loc, scale = res.x[0],res.x[1],res.x[2]
    dist = stats.t(df=dof,loc=loc,scale=scale)
    sim_returns = dist.rvs(size=num_sim)
    # generates sim returns from the t-dist as defined by the parameters solved
    var_t_mle = -np.percentile(sim_returns, alpha*100)
    return var_t_mle, sim_returns, dist, dof, loc, scale

def var_ar1(returns,num_sim=10000, alpha=0.05):
    ar1 = sm.tsa.ARIMA(returns,order=(1,0,0)).fit()
    alpha_ar1 = ar1.params[0]
    resid = ar1.resid
    sigma = np.std(resid)
    sim_returns = []
    for _ in range(num_sim):
        sim_returns.append(alpha_ar1*(returns.iloc[-1])+np.random.normal(0,sigma))
        # simulations of the asset's returns one period ahead
    var_ar1 = np.percentile(sim_returns, alpha*100)
    return var_ar1, sim_returns



# Payoffs are linear and returns are distributed multivariate normal
def delta_normal_VAR(portfolio, prices, alpha=0.05, lambd = 0.94): # also called parametric VaR
    '''
    portfolio: a dataframe with columns 'Stock' and 'Holding'
    prices: a dataframe with columns as stocks and rows as time points
    '''
    current_prices, holdings, portfolio_value = portfolio_calc.calc_portfolio_value(portfolio, prices)
    delta = np.identity(portfolio.shape[0])
    gradient = current_prices/portfolio_value*(delta@holdings)
    prices = prices.loc[:,portfolio['Stock']]
    returns = portfolio_calc.return_calculate(prices).reset_index()
    returns.drop('index',axis=1,inplace=True)
    cov = cov.exp_weighted_cov(returns, lambd)
    VaR = -portfolio_value * stats.norm.ppf(alpha) * np.sqrt(gradient.T @ cov @ gradient)
    return VaR

# Payoffs can be non-linear and returns are distributed multivariate normal
def monte_carlo_VAR(portfolio, prices, nsim=20000, lambd=0.94, alpha=0.05):
    '''
    portfolio: a dataframe with columns 'Stock' and 'Holding'
    prices: a dataframe with columns as stocks and rows as time points
    '''
    current_prices, holdings, _ = portfolio_calc.calc_portfolio_value(portfolio, prices)
    
    prices = prices.loc[:,portfolio['Stock']]
    returns = portfolio_calc.return_calculate(prices).reset_index()
    returns.drop('index',axis=1,inplace=True)
    cov = cov.exp_weighted_cov(returns, lambd)
    
    sim_returns = simulation.multivariate_normal_sim(cov, nsim)
    sim_pv = (current_prices*holdings).T@sim_returns
    VaR = -np.percentile(sim_pv, alpha*100)
    return VaR, sim_pv

def historical_VAR(portfolio, prices, nsim=2000, alpha=0.05):
    '''
    portfolio: a dataframe with columns 'Stock' and 'Holding'
    prices: a dataframe with columns as stocks and rows as time points
    '''
    current_prices, holdings, _ = portfolio_calc.calc_portfolio_value(portfolio, prices)
    
    prices = prices.loc[:,portfolio['Stock']]
    returns = portfolio_calc.return_calculate(prices).reset_index()
    returns.drop('index',axis=1,inplace=True)
    
    sim_returns = returns.sample(nsim, replace=True)
    sim_pv = (current_prices*holdings).T@(sim_returns.T)
    VaR = -np.percentile(sim_pv, alpha*100)
    return VaR, sim_pv



def portfolio_var(prices, holdings, alpha=0.05, dollar=False):
    '''
    prices: a dataframe with columns as stocks and rows as time points
    holdings: 1d np.array corresponding to prices df
    '''
    returns = portfolio_calc.return_calculate(prices).reset_index()
    returns.drop('index',axis=1,inplace=True)
    currentPrices = prices.tail(1).values
    currentValue = holdings * currentPrices
    totValue = np.sum(currentValue)
    currentW = currentValue / totValue
    covar = np.cov(returns.values.T)
    std = np.sqrt(currentW.dot(covar).dot(currentW.T))[0]
    var = -stats.norm.ppf(alpha) * std
    if dollar:
        var *= totValue
    return var
