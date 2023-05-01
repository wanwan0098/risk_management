import numpy as np

def es(returns, alpha=0.05, df=True):
    if df:
        returns.sort_values(inplace=True)
    else: 
        returns = list(returns)
        returns.sort()
    index = round(alpha*len(returns))
    return -np.mean(returns[:index+1])
