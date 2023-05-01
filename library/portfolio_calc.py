import numpy as np
import pandas as pd


# compounded - log
# cross sectional at a given time - discrete
def return_calculate(prices, method='discrete'):
    '''
    input: df without dates
    '''
    price = (prices.shift(-1) / prices).dropna()
    if method.lower() == "discrete":
        return price-1
    elif method.lower() == "log":
        return np.log(price)
    

def return_calculate_adv(prices, method="discrete", datecol="Date"):
    '''
    input: df, with or without dates
    '''
    vars_ = prices.columns
    nVars = len(vars_)
    vars_ = [var for var in vars_ if var != datecol]
    if datecol != 'none':
        if nVars == len(vars_):
            raise ValueError(f"datecol: {datecol} not in DataFrame: {vars_}")
        nVars = nVars - 1

    p = prices[vars_].to_numpy()
    n, m = p.shape
    p2 = np.empty((n-1, m))

    for i in range(n-1):
        for j in range(m):
            p2[i, j] = p[i+1, j] / p[i, j]

    if method.lower() == "discrete":
        p2 = p2 - 1.0
    elif method.lower() == "log":
        p2 = np.log(p2)
    else:
        raise ValueError(f"method: {method} must be in (\"log\",\"discrete\")")

    if datecol != 'none':
        dates = prices[datecol].iloc[1:n].to_numpy()
        out = pd.DataFrame({datecol: dates})
    else:
        out = pd.DataFrame()

    for i in range(nVars):
        out[vars_[i]] = p2[:, i]
    return out

def calc_portfolio_value(portfolio, prices):
    '''
    portfolio: a dataframe with columns 'Stock' and 'Holding'
    prices: a dataframe with columns as stocks and rows as time points
    '''
    dict_current_prices = prices.iloc[-1,:].to_dict()
    portfolio['current_prices'] = portfolio['Stock'].map(dict_current_prices)
    current_prices = portfolio['current_prices'].values
    holdings = portfolio['Holding'].values
    portfolio_value = np.dot(holdings, current_prices)
    return current_prices, holdings, portfolio_value
