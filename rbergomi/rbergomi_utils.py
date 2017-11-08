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

def bs(F, K, V, ϕ=1):
    """
    Returns the Black call price for given forward, strike and variance.
    """
    σ = np.sqrt(V)
    d1 = np.log(F/K) / σ + 0.5 * σ
    d2 = d1 - σ

    C = ϕ * F * norm.cdf(ϕ * d1) - ϕ * K * norm.cdf(ϕ * d2)

    # C = F * norm.cdf( d1) - K * norm.cdf( d2)
    return C

def bsinv(P, F, K, T, ϕ=1):
    """
    Computes implied Black vol from given price, forward, strike and time.
    """
    # Apply at least intrinsic value
    # w = 1
    P = np.maximum(P, np.maximum(ϕ * (F - K), 0))

    def error(σ):
        return bs(F, K, σ**2 * T, ϕ=ϕ) - P
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

def dB2(N,s,ρ=0.5,seed=0):
    """
    Get random numbers for two correlated rBergomi processes, each having
    N paths and s steps.
    """
    np.random.seed(0)
    # Following assumes orthogonal variance components equivalent
    rn = np.random.normal(size=(N,6*s))
    # In what follows the 3 indices correspond to process, factor, hybrid
    # Skew
    dB111 = rn[:,0*s:1*s]
    dB112 = rn[:,1*s:2*s]
    # Curvature
    dB121 = rn[:,4*s:5*s]
    dB122 = rn[:,5*s:6*s]
    # Correlated skew
    dB211 = ρ*rn[:,0*s:1*s]+np.sqrt(1-ρ**2)*rn[:,2*s:3*s]
    dB212 = ρ*rn[:,1*s:2*s]+np.sqrt(1-ρ**2)*rn[:,3*s:4*s]
    # Same curvature
    dB221 = rn[:,4*s:5*s]
    dB222 = rn[:,5*s:6*s]
    # Now prepare 4 hybrid 2D Bms
    dB11 = np.zeros((N,s,2,1)) # α for first
    dB11[:,:,0,0] = dB111
    dB11[:,:,1,0] = dB112
    dB12 = np.zeros((N,s,2,1)) # β for first
    dB12[:,:,0,0] = dB121
    dB12[:,:,1,0] = dB122
    dB21 = np.zeros((N,s,2,1)) # α for second
    dB21[:,:,0,0] = dB211
    dB21[:,:,1,0] = dB212
    dB22 = np.zeros((N,s,2,1)) # β for second
    dB22[:,:,0,0] = dB221
    dB22[:,:,1,0] = dB222
    # Return nicely
    return np.array([[dB11,dB12],[dB21,dB22]])

def dW2(dB2,α1,β1,α2,β2,n):
    """
    Correlate for hybrid scheme.
    """
    dB11 = dB2[0,0]
    dB12 = dB2[0,1]
    dB21 = dB2[1,0]
    dB22 = dB2[1,1]
    # Prepare hybrid choleskys
    cov11 = cov(α1,n)
    cov12 = cov(β1,n)
    cho11 = np.linalg.cholesky(cov11)[np.newaxis,np.newaxis,:,:]
    cho12 = np.linalg.cholesky(cov12)[np.newaxis,np.newaxis,:,:]
    cov21 = cov(α2,n)
    cov22 = cov(β2,n)
    cho21 = np.linalg.cholesky(cov21)[np.newaxis,np.newaxis,:,:]
    cho22 = np.linalg.cholesky(cov22)[np.newaxis,np.newaxis,:,:]
    # And correlate
    dW11 = np.squeeze(np.matmul(cho11,dB11))
    dW12 = np.squeeze(np.matmul(cho12,dB12))
    dW21 = np.squeeze(np.matmul(cho21,dB21))
    dW22 = np.squeeze(np.matmul(cho22,dB22))
    # Return nicely
    return np.array([[dW11,dW12],[dW21,dW22]])
