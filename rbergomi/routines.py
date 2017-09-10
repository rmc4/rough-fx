import numpy as np

def base_run(rbergomi, surface, **kwargs):
    """
    No variance reduction.
    Returns implied vols for surface strikes and maturity.
    Antitheitc sampling specified in rBergomi instance.
    """
    dW1 = rbergomi.dW1()
    dW2 = rbergomi.dW2()
    Y = rbergomi.Y(dW1)
    dB = rbergomi.dB(dW1, dW2, rho = kwargs['rho'])
    V = rbergomi.V(Y, xi = kwargs['xi'], eta = kwargs['eta'])
    S = rbergomi.S(V, dB)
    C = rbergomi.C(S, surface)
    IV = rbergomi.IV(C, surface)
    return IV

def base_put_run(rbergomi, surface, **kwargs):
    """
    No variance reduction.
    Returns implied vols for surface strikes and maturity.
    Antitheitc sampling specified in rBergomi instance.
    """
    dW1 = rbergomi.dW1()
    dW2 = rbergomi.dW2()
    Y = rbergomi.Y(dW1)
    dB = rbergomi.dB(dW1, dW2, rho = kwargs['rho'])
    V = rbergomi.V(Y, xi = kwargs['xi'], eta = kwargs['eta'])
    S = rbergomi.S(V, dB)
    P = rbergomi.P(S, surface)
    IV = rbergomi.IVP(P, surface)
    return IV

def moment_matching(rbergomi, surface, **kwargs):
    """
    Antithetic sampling and moment matching.
    """
    dW1 = rbergomi.dW1()
    dW2 = rbergomi.dW2()
    Y = rbergomi.Y(dW1)
    dB = rbergomi.dB(dW1, dW2, rho = kwargs['rho'])
    V = rbergomi.V(Y, xi = kwargs['xi'], eta = kwargs['eta'])
    S = rbergomi.S(V, dB)
    means = np.mean(S, axis=0)
    S2 = S/means
    C = rbergomi.C(S2, surface)
    IV = rbergomi.IV(C, surface)
    return IV

def conditional_mc(rbergomi, surface, **kwargs):
    """
    Antithetic sampling and conditional Monte Carlo.
    Need to make this more efficient.
    Perpendicular Bm shouldn't be called.
    S is being constructed for repeated paths.
    """
    dW1 = rbergomi.dW1()
    # dW2 = rbergomi.dW2() # Wasted calc.
    Y = rbergomi.Y(dW1)

    dW = dW1[:,:,0]
    N = rbergomi.N
    dB = np.zeros((2 * N, rbergomi.s))
    dB[   :  N,:] = + dW
    dB[  N:2*N,:] = - dW

    # dB = rbergomi.dB(dW1, dW2, rho = 1.) # Wasted calc
    rbergomi.AS = False
    V = np.zeros((2 * N, 1 + rbergomi.s))
    V[:N,:] = rbergomi.V(Y, xi = kwargs['xi'], eta = kwargs['eta'])
    V[N:,:] = rbergomi.V(-Y, xi = kwargs['xi'], eta = kwargs['eta'])

    # V = rbergomi.V(Y, xi = kwargs['xi'], eta = kwargs['eta'])
    S = rbergomi.S2(V, dB, kwargs['rho'])
    C = rbergomi.C3(S, V, surface, kwargs['rho'])
    IV = rbergomi.IV(C, surface)
    return IV

def price_control(rbergomi, surface, **kwargs):
    """
    Antithetic sampling and price control variate.
    """
    dW1 = rbergomi.dW1()
    dW2 = rbergomi.dW2()
    Y = rbergomi.Y(dW1)
    dB = rbergomi.dB(dW1, dW2, rho = kwargs['rho'])
    V = rbergomi.V(Y, xi = kwargs['xi'], eta = kwargs['eta'])
    S = rbergomi.S(V, dB)
    C = rbergomi.C9(S, surface) # Select relevant method
    IV = rbergomi.IV(C, surface)
    return IV

def timer_control(rbergomi, surface, **kwargs):
    """
    Antithetic sampling and TIMER control variate.
    """
    dW1 = rbergomi.dW1()
    dW2 = rbergomi.dW2()
    Y = rbergomi.Y(dW1)
    dB = rbergomi.dB(dW1, dW2, rho = kwargs['rho'])
    V = rbergomi.V(Y, xi = kwargs['xi'], eta = kwargs['eta'])
    S = rbergomi.S(V, dB)
    C = rbergomi.C4(S, V, surface) # Select relevant method
    IV = rbergomi.IV(C, surface)
    return IV

def optimal_run(rbergomi, surface, **kwargs):
    """
    Optimal combination of methods.
    Moment matching did not help.
    MUST PASS THIS EFFICIENCY INTO CONDITIONAL MC ROUTINE!
    """
    dW1 = rbergomi.dW1()
    # dW2 = rbergomi.dW2() # Wasted calc.
    Y = rbergomi.Y(dW1)
    # dB = rbergomi.dB(dW1, dW2, rho = 1.) # Wasted calc

    dW = dW1[:,:,0]
    N = rbergomi.N
    dB = np.zeros((2 * N, rbergomi.s))
    dB[   :  N,:] = + dW
    dB[  N:2*N,:] = - dW
    # dB[2*N:3*N,:] = - dW
    # dB[3*N:   ,:] = - dW

    rbergomi.AS = False
    V = np.zeros((2 * N, 1 + rbergomi.s))
    V[:N,:] = rbergomi.V(Y, xi = kwargs['xi'], eta = kwargs['eta'])
    V[N:,:] = rbergomi.V(-Y, xi = kwargs['xi'], eta = kwargs['eta'])

    S = rbergomi.S2(V, dB, kwargs['rho'])

    C = rbergomi.C6(S, V, surface, kwargs['rho'])
    IV = rbergomi.IV(C, surface)
    return IV

def optimal_scatter(rbergomi, surface, **kwargs):
    """
    Optimal combination of methods.
    Moment matching did not help.
    """
    dW1 = rbergomi.dW1()
    dW2 = rbergomi.dW2() # Wasted calc.
    Y = rbergomi.Y(dW1)
    dB = rbergomi.dB(dW1, dW2, rho = 1.) # Wasted calc
    V = rbergomi.V(Y, xi = kwargs['xi'], eta = kwargs['eta'])
    S = rbergomi.S2(V, dB, kwargs['rho'])
    C = rbergomi.C6_scatter(S, V, surface, kwargs['rho'])
    # IV = rbergomi.IV(C, surface)
    return C

def optimal_run2(rbergomi, surface, **kwargs):
    """
    Optimal combination of methods.
    Appears to not help. Cumsum must not care for path number.
    """
    rbergomi.AS = False # Will run effect from here
    dW1 = rbergomi.dW1()
    Y = rbergomi.Y(dW1)

    # Make AS arrays manually
    Y2 = np.zeros((2 * len(Y[:,0]), len(Y[0,:])))
    Y2[:len(Y[:,0]), :] =  Y
    Y2[len(Y[:,0]):, :] = -Y

    dW12 = np.zeros((2 * len(dW1[:,0,0]), len(dW1[0,:,0])))
    dW12[:len(dW1[:,0,0]), :] =  dW1[:,:,0]
    dW12[len(dW1[:,0,0]):, :] = -dW1[:,:,0]

    # dB = rbergomi.dB(dW1, dW2, rho = 1.) # Wasted calc
    V = rbergomi.V(Y2, xi = kwargs['xi'], eta = kwargs['eta'])
    S = rbergomi.S2(V, dW12, kwargs['rho'])
    C = rbergomi.C6(S, V, surface, kwargs['rho'])
    IV = rbergomi.IV(C, surface)
    return IV

def generate_surface(rbergomi, log_strikes, **kwargs):
    """
    No variance reduction.
    Returns implied vols for surface strikes and maturity.
    Antitheitc sampling specified in rBergomi instance.
    """
    dW1 = rbergomi.dW1()
    dW2 = rbergomi.dW2()
    Y = rbergomi.Y(dW1)
    dB = rbergomi.dB(dW1, dW2, rho = kwargs['rho'])
    V = rbergomi.V(Y, xi = kwargs['xi'], eta = kwargs['eta'])
    S = rbergomi.S(V, dB)
    C = rbergomi.C_from_k(S, log_strikes)
    IV = rbergomi.IV_from_k(C, log_strikes)
    return IV
