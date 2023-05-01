import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.optimize import fsolve

'''
b is the cost of carry
Black Scholes 1973 Option on Equities with no dividends: r=b
Merton 1973 Options on Equities paying a continuous dividend rate q: b=r-q
'''

def black_scholes(S, b, r, T, X, sigma, option):
    d1 = (np.log(S/X)+(b+sigma**2/2)*T)/sigma/T**0.5
    d2 = d1 - sigma*T**0.5
    if option=='call':
        return S*np.exp((b-r)*T)*stats.norm.cdf(d1)-X*np.exp(-r*T)*stats.norm.cdf(d2)
    elif option=='put':
        return X*np.exp(-r*T)*stats.norm.cdf(-d2)-S*np.exp((b-r)*T)*stats.norm.cdf(-d1)
    else:
        return None
    
def implied_vol(S,b,r,T,X,price,option):
    f = lambda sigma: (black_scholes(S,b,r,T,X,sigma,option)-price)
    return fsolve(f, x0 = 0.2, maxfev=10000)[0]


# Greeks calculated using closed-form formulas / GBSM greeks
def d1(S,X,b,sigma,T):
    return (np.log(S/X)+(b+sigma**2/2)*T)/sigma/T**0.5

def d2(S,X,b,sigma,T):
    return d1(S,X,b,sigma,T) - sigma*T**0.5

def calculate_delta(S,b,r,T,X,sigma,option):
    if option=='call':
        return np.exp((b-r)*T)*norm.cdf(d1(S,X,b,sigma,T))
    elif option=='put':
        return np.exp((b-r)*T)*(norm.cdf(d1(S,X,b,sigma,T))-1)

def calculate_gamma(S,b,r,T,X,sigma):
    return norm.pdf(d1(S,X,b,sigma,T))*np.exp((b-r)*T)/(S*sigma*np.sqrt(T))

def calculate_vega(S,b,r,T,X,sigma):
    return S*np.exp((b-r)*T)*norm.pdf(d1(S,X,b,sigma,T))*np.sqrt(T)

def calculate_theta(S,b,r,T,X,sigma,option):
    if option=='call':
        return -S*np.exp((b-r)*T)*norm.pdf(d1(S,X,b,sigma,T))*sigma/2/np.sqrt(T)-(b-r)*S*np.exp((b-r)*T)*norm.cdf(d1(S,X,b,sigma,T))-r*X*np.exp(-r*T)*norm.cdf(d2(S,X,b,sigma,T))
    elif option=='put':
        return -S*np.exp((b-r)*T)*norm.pdf(d1(S,X,b,sigma,T))*sigma/2/np.sqrt(T)+(b-r)*S*np.exp((b-r)*T)*norm.cdf(-d1(S,X,b,sigma,T))+r*X*np.exp(-r*T)*norm.cdf(-d2(S,X,b,sigma,T))

def calculate_rho(S,b,r,T,X,sigma,option):
    if option=='call':
        return T*X*np.exp(-r*T)*norm.cdf(d2(S,X,b,sigma,T))
    elif option=='put':
        return -T*X*np.exp(-r*T)*norm.cdf(-d2(S,X,b,sigma,T))

def calculate_carry_rho(S,b,r,T,X,sigma,option):
    if option=='call':
        return T*S*np.exp((b-r)*T)*norm.cdf(d1(S,X,b,sigma,T))
    elif option=='put':
        return -T*S*np.exp((b-r)*T)*norm.cdf(-d1(S,X,b,sigma,T))
    

# Greeks calculated through the finite difference method
def fd_delta(S,b,r,T,X,sigma,option,d=0.0001):
    d = d*S
    if option=='call':
        return (black_scholes(S+d,b,r,T,X,sigma,'call')-black_scholes(S-d,b,r,T,X,sigma,'call'))/2/d
    elif option=='put':
        return (black_scholes(S+d,b,r,T,X,sigma,'put')-black_scholes(S-d,b,r,T,X,sigma,'put'))/2/d
    
def fd_gamma(S,b,r,T,X,sigma,d=0.0001):
    d = d*S
    return (black_scholes(S+d,b,r,T,X,sigma,'call')+black_scholes(S-d,b,r,T,X,sigma,'call')-2*black_scholes(S,b,r,T,X,sigma,'call'))/d/d

def fd_vega(S,b,r,T,X,sigma,d=0.0001):
    d = d*S
    return (black_scholes(S,b,r,T,X,sigma+d,'call')-black_scholes(S,b,r,T,X,sigma-d,'call'))/2/d

def fd_theta(S,b,r,T,X,sigma,option,d=0.0001):
    d = d*S
    if option=='call':
        res = (black_scholes(S,b,r,T+d,X,sigma,'call')-black_scholes(S,b,r,T-d,X,sigma,'call'))/2/d
        return -res
    elif option=='put':
        res = (black_scholes(S,b,r,T+d,X,sigma,'put')-black_scholes(S,b,r,T-d,X,sigma,'put'))/2/d
        return -res

def fd_rho(S,b,r,T,X,sigma,option,d=0.0001):
    d = d*S
    if option=='call':
        return (black_scholes(S,b+d,r+d,T,X,sigma,'call')-black_scholes(S,b-d,r-d,T,X,sigma,'call'))/2/d
    elif option=='put':
        return (black_scholes(S,b+d,r+d,T,X,sigma,'put')-black_scholes(S,b-d,r-d,T,X,sigma,'put'))/2/d

def fd_carry_rho(S,b,r,T,X,sigma,option,d=0.0001):
    d = d*S
    if option=='call':
        return (black_scholes(S,b+d,r,T,X,sigma,'call')-black_scholes(S,b-d,r,T,X,sigma,'call'))/2/d
    elif option=='put':
        return (black_scholes(S,b+d,r,T,X,sigma,'put')-black_scholes(S,b-d,r,T,X,sigma,'put'))/2/d


# American option pricing
def bt_american_cont(S0,X,T,r,q,sigma,N,option):
    dt = T/N
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    pu = (np.exp((r-q)*dt)-d)/(u-d)
    pd = 1-pu
    discount = np.exp(-r*dt)
    z = 1 if option=='call' else -1
    
    def nNodeFunc(n):
        return (n+1)*(n+2)//2
    def idxFunc(i,j):
        return nNodeFunc(j-1)+i
    nNodes = nNodeFunc(N)
    optionValues = [0.0]*nNodes

    for j in range(N,-1,-1):
        for i in range(j,-1,-1):
            idx = idxFunc(i,j)
            price = S0*u**i*d**(j-i)
            optionValues[idx] = max(0,z*(price-X))
            if j < N:
                optionValues[idx] = max(
                    optionValues[idx],
                    discount*(pu*optionValues[idxFunc(i+1,j+1)] + pd*optionValues[idxFunc(i,j+1)]))
    return optionValues[0]

def bt_american_discrete(S0, X, T, r, sigma, N, option, dividend_times=None, dividend_amounts=None):
    if dividend_times is None or dividend_amounts is None or (len(dividend_amounts)==0) or (len(dividend_times)==0):
        return bt_american_cont(S0, X, T, r, 0, sigma, N, option)
    elif dividend_times[0] > N:
        return bt_american_cont(S0, X, T, r, 0, sigma, N, option)
    
    dt = T/N
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    pu = (np.exp(r*dt)-d)/(u-d)
    pd = 1-pu
    discount = np.exp(-r*dt)
    z = 1 if option == 'call' else -1
    
    def nNodeFunc(n):
        return (n+2)*(n+1)//2
    def idxFunc(i,j):
        return nNodeFunc(j-1)+i
   
    nDiv = len(dividend_times)
    nNodes = nNodeFunc(dividend_times[0])
    optionValues = [0.0]*nNodes

    for j in range(dividend_times[0],-1,-1):
        for i in range(j,-1,-1):
            idx = idxFunc(i,j)
            price = S0*u**i*d**(j-i)       
            
            if j < dividend_times[0]:
                optionValues[idx] = max(0,z*(price-X))
                optionValues[idx] = max(optionValues[idx],
                                        discount*(pu*optionValues[idxFunc(i+1,j+1)] + pd*optionValues[idxFunc(i,j+1)]))
            else:
                no_exer = bt_american_discrete(price-dividend_amounts[0], X,
                                               T-dividend_times[0]*dt, r,
                                               sigma, N-dividend_times[0], option,
                                               [x-dividend_times[0] for x in dividend_times[1:nDiv]],
                                               dividend_amounts[1:nDiv])
                exer = max(0,z*(price-X))
                optionValues[idx] = max(no_exer,exer)

    return optionValues[0]


def fd_greeks_div(S0,X,T,r,sigma,N,dividend_times,dividend_amounts,d=0.0001,delta_only=False):
    d = d*S0
    if delta_only == True:
        delta_call = (bt_american_discrete(S0+d,X,T,r,sigma,N,'call',dividend_times, dividend_amounts)-bt_american_discrete(S0-d,X,T,r,sigma,N,'call',dividend_times, dividend_amounts))/2/d
        delta_put = (bt_american_discrete(S0+d,X,T,r,sigma,N,'put',dividend_times, dividend_amounts)-bt_american_discrete(S0-d,X,T,r,sigma,N,'put',dividend_times, dividend_amounts))/2/d
        return delta_call, delta_put
    delta_call = (bt_american_discrete(S0+d,X,T,r,sigma,N,'call',dividend_times, dividend_amounts)-bt_american_discrete(S0-d,X,T,r,sigma,N,'call',dividend_times, dividend_amounts))/2/d
    delta_put = (bt_american_discrete(S0+d,X,T,r,sigma,N,'put',dividend_times, dividend_amounts)-bt_american_discrete(S0-d,X,T,r,sigma,N,'put',dividend_times, dividend_amounts))/2/d
    gamma_call = (bt_american_discrete(S0+d,X,T,r,sigma,N,'call',dividend_times, dividend_amounts)+bt_american_discrete(S0-d,X,T,r,sigma,N,'call',dividend_times, dividend_amounts)-2*bt_american_discrete(S0,X,T,r,sigma,N,'call',dividend_times, dividend_amounts))/d/d
    gamma_put = (bt_american_discrete(S0+d,X,T,r,sigma,N,'put',dividend_times, dividend_amounts)+bt_american_discrete(S0-d,X,T,r,sigma,N,'put',dividend_times, dividend_amounts)-2*bt_american_discrete(S0,X,T,r,sigma,N,'put',dividend_times, dividend_amounts))/d/d
    vega_call = (bt_american_discrete(S0,X,T,r,sigma+d,N,'call',dividend_times, dividend_amounts)-bt_american_discrete(S0,X,T,r,sigma-d,N,'call',dividend_times, dividend_amounts))/2/d
    vega_put = (bt_american_discrete(S0,X,T,r,sigma+d,N,'put',dividend_times, dividend_amounts)-bt_american_discrete(S0,X,T,r,sigma-d,N,'put',dividend_times, dividend_amounts))/2/d
    theta_call = -(bt_american_discrete(S0,X,T+d,r,sigma,N,'call',dividend_times, dividend_amounts)-bt_american_discrete(S0,X,T-d,r,sigma,N,'call',dividend_times, dividend_amounts))/2/d
    theta_put = -(bt_american_discrete(S0,X,T+d,r,sigma,N,'put',dividend_times, dividend_amounts)-bt_american_discrete(S0,X,T-d,r,sigma,N,'put',dividend_times, dividend_amounts))/2/d
    rho_call = (bt_american_discrete(S0,X,T,r+d,sigma,N,'call',dividend_times, dividend_amounts)-bt_american_discrete(S0,X,T,r-d,sigma,N,'call',dividend_times, dividend_amounts))/2/d
    rho_put = (bt_american_discrete(S0,X,T,r+d,sigma,N,'put',dividend_times, dividend_amounts)-bt_american_discrete(S0,X,T,r-d,sigma,N,'put',dividend_times, dividend_amounts))/2/d
    sens_to_div_call = (bt_american_discrete(S0,X,T,r,sigma,N,'call',dividend_times,[dividend_amounts[0]+d])-bt_american_discrete(S0,X,T,r,sigma,N,'call',dividend_times, [dividend_amounts[0]-d]))/2/d
    sens_to_div_put = (bt_american_discrete(S0,X,T,r,sigma,N,'put',dividend_times,[dividend_amounts[0]+d])-bt_american_discrete(S0,X,T,r,sigma,N,'put',dividend_times, [dividend_amounts[0]-d]))/2/d
    return delta_call,delta_put,gamma_call,gamma_put,vega_call,vega_put,theta_call,theta_put,rho_call,rho_put,sens_to_div_call,sens_to_div_put
