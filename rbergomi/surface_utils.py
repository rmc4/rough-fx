import pandas as pd
import numpy as np
from scipy.stats import norm

def csv_import(currency, date):
    """
    Function for importing Bloomberg FX volatility data from a csv file for
    a given date. Currency should be passed as 'GBPUSD' and date as '2017-01-31'.
    """
    # tab delimiter used since coming from excel
    df = pd.read_csv('data/'+ currency +'-'+ date +'.csv', header=None, sep='\s+')
    deltas = np.array(df.loc[0,1:])
    tenors = np.array(df.loc[1:,0])
    vols = np.array(df.loc[1:,1:]).astype(float)
    return pd.DataFrame(vols, index=tenors, columns=deltas)

def tenors_yearfracs(tenors):
    """
    Convert numpy array from tenors to year fractions.
    """
    tenor_dict = {'D': 1./312,
                  'W': 1./52,
                  'M': 1./12,
                  'Y': 1.}

    yearfracs = np.zeros(len(tenors))
    i = 0
    for tenor in tenors:
        key = tenor[-1]
        multiple = int(tenor[:-1])
        yearfracs[i] = multiple * tenor_dict[key]
        i += 1
    return yearfracs

def delta_to_put(delta, maturity, vol):
    """
    Converts a string delta, e.g. 45C, to put deltas, N(-d1). Needs to be
    vectorised for use over NumPy arrays
    """
    if delta[-1] == 'C':
        put_delta = 1 - float(delta[:-1])/100
    if delta == 'ATM':
        put_delta = norm.cdf(- vol * np.sqrt(maturity)/2)
    if delta[-1] == 'P':
        put_delta = float(delta[:-1])/100
    return put_delta

def put_to_forward(put_delta, maturity, vol):
    """
    Converts put delta, N(-d1), to forward delta, N(-d2).
    """
    d1 = - norm.ppf(put_delta)
    d2 = d1 - vol * np.sqrt(maturity)
    return norm.cdf(-d2)

def put_delta_to_strike(put_delta, vol, maturity):
    """
    Converts put delta, N(-d1), to strike for given volatility and maturity.
    This follows easily from the definition of d1.
    """
    d1 = - norm.ppf(put_delta)
    log = d1 * vol * np.sqrt(maturity) - 0.5 * vol**2 * maturity
    return 1./np.exp(log)

def strike_to_log_strike(strike):
    """
    Converts strike, K, to log-strike, k = log(K/F).
    """
    return np.log(strike / 1.)

def price(F, K, T, s):
    """
    generalise with weight.
    Returns the forward BS price for given forward, strike, time and
    volatility
    """
    d1 =(np.log(F/K) + 0.5 * s**2 * T)/(s * np.sqrt(T))
    d2 = d1 - s * np.sqrt(T)
    P = F * norm.cdf(d1) - K * norm.cdf(d2)
    return P
