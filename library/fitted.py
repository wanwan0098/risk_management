from scipy import stats
from scipy.optimize import minimize

# residuals follow t-dist
def t_err_MLE(x, y):
    def t_ll(params):
        yhat = params[0]+params[1]*x
        ll = stats.t.logpdf(y-yhat, df=params[2], loc=0, scale=params[3]).sum() # x's, shape(dof), mean, scale
        return -ll
    t_res = minimize(t_ll, x0=(1, 1, 1, 1), method = 'Nelder-Mead') # alpha, beta, dof, scale
    # try method = 'Nelder-Mead' if optimization fails
    intercept, slope = t_res.x[0], t_res.x[1]
    errors = y - (intercept+slope*x)
    return errors, intercept, slope

# residuals follow Normal dist
# note: fitting the data using MLE given the assumption that residuals are normally dist renders the same
# parameter results as fitting the data in the context of OLS
def normal_err_MLE(x, y):
    def normal_ll(params):
        yhat = params[0]+params[1]*x
        ll = stats.norm.logpdf(y-yhat, scale=params[2], loc=0).sum()
        return -ll
    n_res = minimize(normal_ll, x0=(1,1,1))
    return n_res
