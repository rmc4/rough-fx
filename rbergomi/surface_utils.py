import numpy as np
import xlwings as xw
from scipy.stats import norm

def xl_import(currency, date, close = True):
    """
    Function for importing Bloomberg FX volatility data from an xlsx file for
    a given date. Currency should be passed as 'GBPUSD' and data as '20170131'.
    """
    # Obtain path of .xlsx file
    install_path = '/Users/ryanmccrickerd/desktop/phd/2016-17' # Send to yaml config
    # file_path = install_path + '/rbergomi/data/market/' + date + '_' + currency + '.xlsx'
    file_path = install_path + '/turbo_rbergomi/data/vr_tests/' + date + '_' + currency + '.xlsx'
    # file_path = install_path + '/turbo_rbergomi/data/' + date + '_' + currency + '.xlsx'

    # Instantiate xlwings object
    wb = xw.Book(file_path)
    sht = wb.sheets['Sheet1']

    # Assign meaningful data in sheet
    data = sht.range('A1').expand().value

    if close:
        wb.close() # Close workbook without saving
    return np.array(data)

def tenors_to_yearfracs(tenors):
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

def delta_to_put(delta, maturity, volatility, scale=100):
    """
    Converts a string delta, e.g. 45C, to put deltas, N(-d1). Needs to be
    vectorised for use over NumPy arrays
    """
    if delta[-1] == 'C':
        put_delta = 1 - float(delta[:-1])/scale
    if delta == 'ATM':
        put_delta = norm.cdf(- volatility * np.sqrt(maturity)/2)
    if delta[-1] == 'P':
        put_delta = float(delta[:-1])/scale
    return put_delta

def put_to_forward(put_delta, maturity, volatility):
    """
    Converts put delta, N(-d1), to forward delta, N(-d2).
    """
    d1 = - norm.ppf(put_delta) # PPF: percentage point function, CDF inverse
    d2 = d1 - volatility * np.sqrt(maturity)
    return norm.cdf(-d2)

def put_delta_to_strike(forward, put_delta, volatility, maturity):
    """
    Converts put delta, N(-d1), to strike for given volatility and maturity.
    This follows easily from the definition of d1.
    """
    d1 = - norm.ppf(put_delta)
    log = d1 * volatility * np.sqrt(maturity) - 0.5 * volatility**2 * maturity
    return forward/np.exp(log)

def strike_to_log_strike(forward, strike):
    """
    Converts strike, K, to log-strike, k = log(K/F).
    """
    return np.log(strike / forward)

def call_price(F, K, T, s):
    """
    Returns the forward BS price for given forward, strike, time and
    volatility
    """
    d1 =(np.log(F/K) + 0.5 * s**2 * T)/(s * np.sqrt(T))
    d2 = d1 - s * np.sqrt(T)
    C = F * norm.cdf(d1) - K * norm.cdf(d2)
    return C

def put_price(F, K, T, s):
    """
    Returns the forward BS price for given forward, strike, time and
    volatility
    """
    d1 =(np.log(F/K) + 0.5 * s**2 * T)/(s * np.sqrt(T))
    d2 = d1 - s * np.sqrt(T)
    P = K * norm.cdf(-d2) - F * norm.cdf(-d1)
    return P
