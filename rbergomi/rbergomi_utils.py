import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def g(x, a):
    """
    Kernel for rBergomi TBSS process.
    """
    return x**a

def h(x, a, b):
    """
    Gamma kernel for rGamma TBSS process.
    """
    return x**a * np.exp(-b*x)

def L(x, b):
    """
    Gamma kernel for rGamma TBSS process.
    """
    return np.exp(-b*x)

def b(k, a):
    """
    Optimal discretisation of TBSS to minimise error, p. 9.
    """
    return ((k**(a+1)-(k-1)**(a+1))/(a+1))**(1/a)

def b2(k, a):
    """
    Optimal discretisation of TBSS to minimise error, p. 9.
    """
    return ((k**(a+1)-(k-1)**(a+1))/(a+1))**(1/a)

def cov(a, n):
    """
    Covariance matrix for given alpha and granularity. Taken from p. 14 of
    hybrid scheme (2015), assumes kappa = 1.
    """
    cov = np.array([[0.,0.],[0.,0.]])
    cov[0,0] = 1./n
    cov[0,1] = 1./((1.*a+1) * n**(1.*a+1))
    cov[1,1] = 1./((2.*a+1) * n**(2.*a+1))
    cov[1,0] = cov[0,1]
    return cov

def bs(F, K, V):
    """
    Returns the Black call price for given forward, strike and variance.
    """
    s = np.sqrt(V)
    d1 = np.log(F/K) / s + 0.5 * s
    d2 = d1 - s

    # ADDING PUT FUNCTIONALITY 1 OF 3
    w = 2 * (K > 1.0) - 1
    # w = 1
    C = w * F * norm.cdf(w * d1) - w * K * norm.cdf(w * d2)

    # C = F * norm.cdf( d1) - K * norm.cdf( d2)
    return C

def blsimpv(P, F, K, T):
    """
    Computes implied Black vol from given price, forward, strike and time.
    """
    # Apply at least intrinsic value
    # PUT OPTION ADDITION 2 OF 3
    w = 2 * (K > 1.0) - 1
    # w = 1
    P = np.maximum(P, np.maximum(w * (F - K), 0))

    def error(s):
        return bs(F, K, s**2 * T) - P
    # if P == np.maximum(F - K, 0):
    #     result = 0
    # else:
    result = brentq(error, 1e-5, 1e+5)
    return result

def rmse(actual, implied):
    """
    Basic RMSE computation for actual vs. implied volatility surfaces.
    """
    #rmse = np.sqrt(np.mean(((implied - actual) / actual)**2))
    rmse = np.sqrt(np.mean((implied - actual)**2))
    return rmse
