import numpy as np
import pandas as pd
from rbergomi.rbergomi_utils import *

class rBergomi(object):
    """
    Class for generating paths of the rBergomi model.

    Integral equations for reference:
    Y(t) := sqrt(2a + 1) int 0,t (t - u)^a dW(u)
    V(t) := xi exp(eta Y - 0.5 eta^2 t^(2a + 1))
    S(t) := S0 int 0,t sqrt(V) dB(u) - 0.5 V du
    """
    def __init__(self, n=156, N=1024, T=1.0):
        """
        Constructor for class.
        """
        # Basic assignments.
        self.n = n # Steps per year
        self.N = N # Paths
        self.T = T # Maturity
        self.dt = 1.0/n # Step size
        self.s = int(n*T) # Steps
        self.t = np.linspace(0,T,1+self.s)[np.newaxis,:] # Time grid

    def dW(self, α=0.4, β=-0.4, seed=0):
        """
        .
        """
        self.α = α
        self.β = β
        s = self.s

        # Store required covariance matrices
        cov1 = cov(α, self.n)
        cov2 = cov(β, self.n)
        chol1 = np.linalg.cholesky(cov1)[np.newaxis,np.newaxis,:,:]
        chol2 = np.linalg.cholesky(cov2)[np.newaxis,np.newaxis,:,:]


        fn = 'sobol/'+str(seed)+'-'+str(self.N)+'-'+str(4*s)+'.csv'
        random_numbers = np.array(pd.read_csv(fn))

        # Obviously generalise
        dB11 = random_numbers[:,0*s:1*s]
        dB12 = random_numbers[:,1*s:2*s]
        dB21 = random_numbers[:,2*s:3*s]
        dB22 = random_numbers[:,3*s:4*s]

        # Prepare for operations
        dB1 = np.zeros((self.N,s,2,1))
        dB2 = np.zeros((self.N,s,2,1))

        dB1[:,:,0,0] = dB11
        dB1[:,:,1,0] = dB12
        dB2[:,:,0,0] = dB21
        dB2[:,:,1,0] = dB22

        # Finally, correlate in C-layer
        dW1 = np.squeeze(np.matmul(chol1,dB1))
        dW2 = np.squeeze(np.matmul(chol2,dB2))

        dW = np.zeros((self.N,s,2,2))
        dW[:,:,:,0] = dW1
        dW[:,:,:,1] = dW2

        return dW

    # Should promote this for two dimensions given α, β use
    def Y(self, dW, α):
        """
        Constructs Volterra process from appropriately
        correlated 2d Brownian increments.
        """
        Y1 = np.zeros((self.N, 1 + self.s)) # Exact integral
        Y2 = np.zeros((self.N, 1 + self.s)) # Riemann sum

        # Construct Y1 through exact integral
        # for i in range(1 + self.s):
        # Use np.cumsum here? - must time this
        for i in np.arange(1, 1 + self.s, 1): # See (3.6)
            Y1[:,i] += dW[:,i-1,1] # Assumes kappa = 1

        # Construct arrays for convolution
        Γ = np.zeros(1 + self.s) # Gamma
        for k in np.arange(2, 1 + self.s, 1): # Assumes kappa = 1
            Γ[k] = g(b(k, α)/self.n, α)

        Ξ = dW[:,:,0] # Xi

        # Initialise convolution result, GX
        ΓΞ = np.zeros((self.N, len(Ξ[0,:]) + len(Γ) - 1))

        # Compute convolution, FFT not used for small n
        # Not able to compute all paths in C-layer
        for i in range(self.N):
            ΓΞ[i,:] = np.convolve(Γ, Ξ[i,:])

        # Extract appropriate part of convolution
        Y2 = ΓΞ[:,:1 + self.s]

        # Finally contruct and return full process
        Y = np.sqrt(2 * α + 1) * (Y1 + Y2)
        # Y = (Y1 + Y2)
        return Y

    # def dW2(self):
    #     """
    #     Obtain orthogonal increments.
    #     """
    #     return np.random.randn(self.N, self.s) * np.sqrt(self.dt)
    #
    # def dB(self, dW1, dW2, rho = 0.0):
    #     """
    #     Method for obtaining price Brownian increments, dB, from variance, dW.
    #     """
    #     # Take variance increments from argument
    #     dW10 = dW1[:,:,0]
    #
    #     # Act accorinding to AV token
    #     if self.AS:
    #         # Now correlate 4 cases appropriately
    #         dB1 = rho * dW10 + np.sqrt(1 - rho**2) * dW2 # + + use + Y
    #         dB2 = rho * dW10 - np.sqrt(1 - rho**2) * dW2 # + - use + Y
    #         dB3 = - dB2                                  # - + use - Y
    #         dB4 = - dB1                                  # - - use - Y
    #
    #         N = self.N
    #         dB = np.zeros((4 * N, self.s))
    #         dB[   :  N,:] = + dB1
    #         dB[  N:2*N,:] = + dB2
    #         dB[2*N:3*N,:] = - dB2
    #         dB[3*N:   ,:] = - dB1
    #     else:
    #         # Just the single construction
    #         dB = rho * dW10 + np.sqrt(1 - rho**2) * dW2
    #
    #     # Assign for later use
    #     self.rho = rho
    #     return dB

    # Yes should raise dimens
    def V(self, Yα, Yβ, ξ=1.0, ζ=1.0, η=1.0):
        """
        rBergomi variance process.
        SHOULD ALSO WRITE INTEGRATED PROCESS METHOD FOR EFFICIENT LATER USE.
        """

        self.ξ = ξ
        self.ζ = ζ
        self.η = η

        α = self.α
        β = self.β
        t = self.t

        Vα = np.exp(ζ*Yα - 0.5*ζ**2 * t**(2*α+1))
        Vβ = np.exp(η*Yβ - 0.5*η**2 * t**(2*β+1))

        V = ξ * Vα * Vβ
        return V

    def S(self, V, dB):
        """
        rBergomi price process.
        """
        dt = self.dt

        # Construct non-anticipative Riemann increments
        increments = np.sqrt(V[:,:-1]) * dB - 0.5 * V[:,:-1] * dt

        # Cumsum is actually a little slower than Python loop. Not terribly
        integral = np.cumsum(increments, axis = 1)

        S = np.zeros_like(V)
        S[:,0] = 1.
        S[:,1:] = np.exp(integral)
        return S

    # But still used in solver at moment!
    # Should pass Surface object here rather than maturities and strikes

    def surface(self, S, surf):
        """
        Provides the implied Black volatility surface for every option
        implicitely in a Surface object.
        """
        vec_bsinv = np.vectorize(bsinv)

        indices = (surf.maturities * self.n).astype(int)
        ST = S[:,indices][:,:,np.newaxis]
        K = np.array(surf.strikes())[np.newaxis,:,:]
        Δ = np.array(surf.put_deltas())
        T = surf.maturities[:,np.newaxis]

        call_payoffs = np.maximum(ST - K,0) - (1-Δ)*(ST - 1)
        call_prices = np.mean(call_payoffs, axis=0)
        call_vols = vec_bsinv(call_prices, 1., np.squeeze(K), T, ϕ=1)

        put_payoffs = np.maximum(K - ST,0) + Δ*(ST - 1)
        put_prices = np.mean(put_payoffs, axis=0)
        put_vols = vec_bsinv(put_prices, 1., np.squeeze(K), T, ϕ=-1)

        vols = (call_vols + put_vols) / 2

        return pd.DataFrame(vols, index=surf.tenors, columns=surf.deltas)

    # def surface(self, S, maturities, log_strikes):
    #     """
    #     Provides the implied Black volatility surface for every option
    #     implicitely in a Surface object.
    #     """
    #     surface = np.zeros_like(log_strikes)
    #
    #     M = maturities
    #     k = log_strikes
    #     K = np.exp(k)
    #
    #     loc = (M * self.n).astype(int) # Tidy this up..
    #
    #     # Extract distribution of S at slices we care about
    #     ST = np.zeros((len(S[:,0]), len(loc)))
    #     j = 0
    #     for i in loc:
    #         ST[:,j] = np.squeeze(S[:,i]) # Not sure why squeeze is required..
    #         j += 1
    #
    #     # Diabolical. Simplify through broadcasting
    #     # Place these functions in utils
    #     C = np.zeros_like(K)
    #     for j in range(len(C[0,:])):
    #         for i in range(len(C[:,0])):
    #             w = 2 * (K[i,j] > 1.0) - 1 # CHANGE 4 OF 4
    #             C[i,j] = np.mean(np.maximum(w*(ST[:,i] - K[i,j]), 0))
    #
    #     # Solver doesn't appear to accept broadcasting
    #     for j in range(len(loc)):
    #         for i in range(len(k[0,:])):
    #             surface[j,i] = blsimpv(C[j,i], 1., K[j,i], M[j]) # Spot = 1
    #
    #     return surface

    def surface2(self, C, maturities, log_strikes):
        """
        Provides the implied Black volatility surface for every option
        implicitely in a Surface object.
        """
        surface = np.zeros_like(log_strikes)

        M = maturities
        k = log_strikes
        K = np.exp(k)

        loc = (M * self.n).astype(int) # Tidy this up..

        # Extract distribution of S at slices we care about
        # ST = np.zeros((len(S[:,0]), len(loc)))
        # j = 0
        # for i in loc:
        #     ST[:,j] = np.squeeze(S[:,i]) # Not sure why squeeze is required..
        #     j += 1

        # Diabolical. Simplify through broadcasting
        # Place these functions in utils
        # C = np.zeros_like(K)
        # for j in range(len(C[0,:])):
        #     for i in range(len(C[:,0])):
        #         C[i,j] = np.mean(np.maximum(ST[:,i] - K[i,j], 0))

        # Solver doesn't appear to accept broadcasting
        for j in range(len(loc)):
            for i in range(len(k[0,:])):
                surface[j,i] = blsimpv(C[j,i], 1., K[j,i], M[j]) # Spot = 1

        return surface

    # This isn't really particular to the class. Leave in utils?
    def rmse(self, actual_surface, rbergomi_surface):
        """
        Returns the RMSE between data and simulated volatility surfaces.
        """
        return rmse(actual_surface, rbergomi_surface)

    def P(self, S, surface):
        """
        Same as below but for puts. To test if better at low strike.
        """
        K = np.exp(np.array(surface._log_strike_surface())) # Assumes S0 = 1
        M = (surface._maturities * self.n).astype(int) # Location of expiries

        P = np.zeros((len(S[:,0]), len(K[:,0]), len(K[0,:])))
        for i in range(len(M[:,0])):
            ST = S[:,M[i,0]][:,np.newaxis]
            KT = K[i,:][np.newaxis,:]
            P[:,i,:] = np.maximum(KT - ST, 0) # Note broadcasting

        # Return only expectation and standard deviation
        e = np.mean(P, axis = 0)
        s = np.std(P, axis = 0) / np.sqrt(len(P[:,0,0])) # Division by sqrt paths
        return np.transpose([e,s], [1,2,0]) # Order axes naturally

    def C(self, S, surface):
        """
        Returns call option payoffs from a price process and surface. The
        resulting cube takes dimensions N x M x K = paths x expiries x strikes
        This is required to obtain error bounds on the implied call prices
        """
        K = np.exp(np.array(surface._log_strike_surface())) # Assumes S0 = 1
        M = (surface._maturities * self.n).astype(int) # Location of expiries

        C = np.zeros((len(S[:,0]), len(K[:,0]), len(K[0,:])))
        for i in range(len(M[:,0])):
            ST = S[:,M[i,0]][:,np.newaxis]
            KT = K[i,:][np.newaxis,:]

            # ADDING PUT FUNCTIONALITY 3 OF 3
            w = 2 * (KT > 1.0) - 1
            # w = 1
            C[:,i,:] = np.maximum(w * (ST - KT), 0) # Note broadcasting

        # Return only expectation and standard deviation
        e = np.mean(C, axis = 0)
        s = np.std(C, axis = 0) / np.sqrt(len(C[:,0,0])) # Division by sqrt paths
        return np.transpose([e,s], [1,2,0]) # Order axes naturally

    def C2(self, V, surface):
        """
        Returns BS call option prices conditional on the variance paths.
        CRITICALLY, this assumes no correlation between driving BMs. The
        resulting cube takes dimensions N x M x K = paths x expiries x strikes
        This is required to obtain error bounds on the implied call prices
        """
        K = np.exp(np.array(surface._log_strike_surface())) # Assumes S0 = 1
        M = (surface._maturities * self.n).astype(int) # Location of expiries
        T = surface._maturities

        # Construct integrated variances for conditional vols
        # More efficient to use variances directly in BS rather than vols
        sigma = np.sqrt(np.cumsum(V, axis = 1) * self.dt / self.t)

        C = np.zeros((len(V[:,0]), len(K[:,0]), len(K[0,:])))
        for i in range(len(M[:,0])):
            sigmaT = sigma[:,M[i,0]][:,np.newaxis]
            KT = K[i,:][np.newaxis,:]
            # C[:,i,:] = np.maximum(ST - KT, 0) # Note broadcasting
            C[:,i,:] = bs(1.0, KT, sigmaT**2 * T) # Note broadcasting

        # Return only expectation and standard deviation
        e = np.mean(C, axis = 0)
        s = np.std(C, axis = 0) / np.sqrt(len(C[:,0,0])) # Division by sqrt paths
        return np.transpose([e,s], [1,2,0]) # Order axes naturally

    def C3(self, S, V, surface, rho):
        """
        Returns BS call option prices conditional on the variance and
        PARALLEL price paths. This assumes S and V are already appropriately
        correlated. Can later be made more efficient. The resulting cube takes
        dimensions N x M x K = paths x expiries x strikes. This is required to
        obtain error bounds on the implied call prices.
        """
        K = np.exp(np.array(surface._log_strike_surface())) # Assumes S0 = 1
        M = (surface._maturities * self.n).astype(int) # Location of expiries
        T = surface._maturities

        # Construct integrated variances for conditional vols
        # More efficient to use variances directly in BS rather than vols
        sigma = np.sqrt(np.cumsum(V, axis = 1) * self.dt / self.t)

        C = np.zeros((len(V[:,0]), len(K[:,0]), len(K[0,:])))
        for i in range(len(M[:,0])):
            # Construction of sigma is sexy
            # Collapses to payoff in rho = +-1 case
            # So evaluation of BS completely redundant
            sigmaT = np.sqrt((1 - rho**2)) * sigma[:,M[i,0]][:,np.newaxis]

            ST = S[:,M[i,0]][:,np.newaxis]
            KT = K[i,:][np.newaxis,:]
            # C[:,i,:] = np.maximum(ST - KT, 0) # Note broadcasting
            C[:,i,:] = bs(ST, KT, sigmaT**2 * T) # Note broadcasting

        # Return only expectation and standard deviation
        e = np.mean(C, axis = 0)
        s = np.std(C, axis = 0) / np.sqrt(len(C[:,0,0])) # Division by sqrt paths
        return np.transpose([e,s], [1,2,0]) # Order axes naturally

    def C4(self, S, V, surface):
        """
        First attempt with control variates. We follow Bergomi (2015) but
        fix the log S QV budget at the known maximum so that the control doesn't
        have to change (incurring jumps at unknown random stopping times). We
        choose an optimum scaling factor to reduce variance. Here the payoff is
        computed and the BS formula used on full paths. Next we will try
        combining a control variant with C3 parallel BS computations. The
        resulting cube takes dimensions N x M x K = paths x expiries x strikes.
        This is required to obtain error bounds on the implied call prices.
        """
        K = np.exp(np.array(surface._log_strike_surface())) # Assumes S0 = 1
        M = (surface._maturities * self.n).astype(int) # Location of expiries
        T = surface._maturities

        # Construct integrated variances for conditional vols
        # More efficient to use variances directly in BS rather than vols
        sigma = np.sqrt(np.cumsum(V, axis = 1) * self.dt / self.t)

        # Compute maximum QV of log S achieved
        # This should be at EVERY desired time slice!
        maxQV = np.amax(sigma[:,-1])**2 * T[-1]+0.0001

        C = np.zeros((len(V[:,0]), len(K[:,0]), len(K[0,:])))
        D = np.zeros((len(V[:,0]), len(K[:,0]), len(K[0,:]))) # Control
        for i in range(len(M[:,0])):
            # Construction of sigma is sexy
            # Collapses to payoff in rho = +-1 case
            # So evaluation of BS completely redundant
            # sigmaT = np.sqrt((1 - rho**2)) * sigma[:,M[i,0]][:,np.newaxis]
            QVT =  sigma[:,M[i,0]][:,np.newaxis]**2 * T[-1]

            ST = S[:,M[i,0]][:,np.newaxis]
            KT = K[i,:][np.newaxis,:]
            # C[:,i,:] = np.maximum(ST - KT, 0) # Note broadcasting
            # C[:,i,:] = blsprice(ST, KT, T, sigmaT) # Note broadcasting
            w = 2 * (KT > 1.0) - 1
            C[:,i,:] = np.maximum(w*(ST - KT), 0) # Note broadcasting
            D[:,i,:] = bs(ST, KT, maxQV - QVT) # Control

        # Now must compute optimal control constant
        # Must be careful with broadcasting for strikes
        # Assumes single maturity
        c = np.zeros((1,1,len(K[0,:])))
        for i in range(len(K[0,:])):
            covMat = np.cov(C[:,0,i], D[:,0,i])
            c[0,0,i] = - covMat[0,1] / covMat[1,1]

        # Expectation of control
        eCV = bs(1.0, KT, maxQV) # Broadcast to K shape

        # Finally make new sample
        E = C + c*(D - eCV)

        # Should instead return block and report upon it
        # Return only expectation and standard deviation
        e = np.mean(E, axis = 0)
        s = np.std(E, axis = 0) / np.sqrt(len(E[:,0,0])) # Division by sqrt paths
        return np.transpose([e,s], [1,2,0]) # Order axes naturally

    def C5(self, S, S2, V, surface, rho):
        """
        Attempting to mix control variates with BS evaluation.
        We follow Bergomi (2015) but
        fix the log S QV budget at the known maximum so that the control doesn't
        have to change (incurring jumps at unknown random stopping times). We
        choose an optimum scaling factor to reduce variance. Here the payoff is
        computed and the BS formula used on full paths. Next we will try
        combining a control variant with C3 parallel BS computations. The
        resulting cube takes dimensions N x M x K = paths x expiries x strikes.
        This is required to obtain error bounds on the implied call prices.
        """
        K = np.exp(np.array(surface._log_strike_surface())) # Assumes S0 = 1
        M = (surface._maturities * self.n).astype(int) # Location of expiries
        T = surface._maturities

        # Construct integrated variances for conditional vols
        # More efficient to use variances directly in BS rather than vols
        sigma = np.sqrt(np.cumsum(V, axis = 1) * self.dt / self.t)

        # Compute maximum QV of log S achieved
        # This should be at EVERY desired time slice!
        maxQV = np.amax(sigma[:,-1])**2 * T[-1]

        C = np.zeros((len(V[:,0]), len(K[:,0]), len(K[0,:])))
        D = np.zeros((len(V[:,0]), len(K[:,0]), len(K[0,:]))) # Control
        for i in range(len(M[:,0])):
            # Construction of sigma is sexy
            # Collapses to payoff in rho = +-1 case
            # So evaluation of BS completely redundant
            sigmaT = np.sqrt((1 - rho**2)) * sigma[:,M[i,0]][:,np.newaxis]
            QVT = sigma[:,M[i,0]][:,np.newaxis]**2 * T[-1]

            ST = S[:,M[i,0]][:,np.newaxis]
            ST2 = S2[:,M[i,0]][:,np.newaxis]
            KT = K[i,:][np.newaxis,:]
            # C[:,i,:] = np.maximum(ST - KT, 0) # Note broadcasting
            # Must pass only parallel ST into this method!!
            C[:,i,:] = bs(ST, KT, sigmaT**2 * T) # Note broadcasting
            # C[:,i,:] = np.maximum(ST - KT, 0) # Note broadcasting
            D[:,i,:] = bs(ST2, KT, maxQV - QVT) # Control

        # E[D] did not give eCV when I tried using only parallel part in
        # control -- have to think about this...

        # Now must compute optimal control constant
        # Must be careful with broadcasting for strikes
        # Assumes single maturity
        c = np.zeros((1,1,len(K[0,:])))
        for i in range(len(K[0,:])):
            covMat = np.cov(C[:,0,i], D[:,0,i])
            c[0,0,i] = - covMat[0,1] / covMat[1,1]

        # Expectation of control
        eCV = bs(1.0, KT, maxQV) # Broadcast to K shape

        # Finally make new sample
        E = C + c*(D - eCV)

        # Should instead return block and report upon it
        # Return only expectation and standard deviation
        e = np.mean(E, axis = 0)
        s = np.std(E, axis = 0) / np.sqrt(len(E[:,0,0])) # Division by sqrt paths
        return np.transpose([e,s], [1,2,0]) # Order axes naturally

    def C6(self, S, V, surface, rho):
        """
        Attempting to mix PARALLEL control variates with BS evaluation.
        We follow Bergomi (2015) but
        fix the log S QV budget at the known maximum so that the control doesn't
        have to change (incurring jumps at unknown random stopping times). We
        choose an optimum scaling factor to reduce variance. Here the payoff is
        computed and the BS formula used on full paths. Next we will try
        combining a control variant with C3 parallel BS computations. The
        resulting cube takes dimensions N x M x K = paths x expiries x strikes.
        This is required to obtain error bounds on the implied call prices.
        """
        K = np.exp(np.array(surface._log_strike_surface())) # Assumes S0 = 1
        M = (surface._maturities * self.n).astype(int) # Location of expiries
        T = surface._maturities

        # Construct integrated variances for conditional vols
        # More efficient to use variances directly in BS rather than vols
        sigma = np.sqrt(np.cumsum(V, axis = 1) * self.dt / self.t)

        # Compute maximum QV of log S achieved
        # This should be at EVERY desired time slice!
        maxQV = np.amax(sigma[:,-1])**2 * T[-1]

        C = np.zeros((len(V[:,0]), len(K[:,0]), len(K[0,:])))
        D = np.zeros((len(V[:,0]), len(K[:,0]), len(K[0,:]))) # Control
        eCV = np.zeros((len(D[0,:,0]), len(D[0,0,:])))
        for i in range(len(M[:,0])):
            # Construction of sigma is sexy
            # Collapses to payoff in rho = +-1 case
            # So evaluation of BS completely redundant
            sigmaT = np.sqrt((1 - rho**2)) * sigma[:,M[i,0]][:,np.newaxis]
            QVT = sigma[:,M[i,0]][:,np.newaxis]**2 * T[i,0]
            maxQVT = np.max(QVT)

            ST = S[:,M[i,0]][:,np.newaxis]
            KT = K[i,:][np.newaxis,:]
            # C[:,i,:] = np.maximum(ST - KT, 0) # Note broadcasting
            # Must pass only parallel ST into this method!!
            C[:,i,:] = bs(ST, KT, sigmaT**2 * T[i,0]) # Note broadcasting
            # C[:,i,:] = np.maximum(ST - KT, 0) # Note broadcasting

            # if np.abs(rho) > 0.1:
            D[:,i,:] = bs(ST, KT, rho**2*(maxQVT - QVT)) # Control

            # Just need to broadcast this properly
            # Expectation of control
            eCV[i,:] = bs(1.0, KT, rho**2 * maxQVT) # Broadcast to K shape

        # E[D] did not give eCV when I tried using only parallel part in
        # control -- have to think about this...

        # Now must compute optimal control constant
        # Must be careful with broadcasting for strikes
        # Assumes single maturity
        # if np.abs(rho) > 0.1:
        c = np.zeros((1,len(K[:,0]),len(K[0,:])))
        for j in range(len(K[:,0])):
            for i in range(len(K[0,:])):
                covMat = np.cov(C[:,j,i], D[:,j,i])
                c[0,j,i] = - covMat[0,1] / covMat[1,1]

        # Expectation of control
        # eCV = bs(1.0, KT, rho**2 * maxQVT) # Broadcast to K shape
        # eCV = np.zeros((len(D[0,:,0]), len(D[0,0,:])))
        # for i in range(len(M[:,0])):
        #     eCV[i,:] = np.mean(D[:,i,:], axis = 0) # Broadcast to K shape
        #
        # # Just need to broadcast this properly
        # # Expectation of control
        # eCV = bs(1.0, KT, rho**2 * maxQVT) # Broadcast to K shape

        # Finally make new sample
        if np.abs(rho) > 0.1:
            E = C + c*(D - eCV)
        else:
            E = C

        # Should instead return block and report upon it
        # Return only expectation and standard deviation
        e = np.mean(E, axis = 0)
        s = np.std(E, axis = 0) / np.sqrt(len(E[:,0,0])) # Division by sqrt paths
        return np.transpose([e,s], [1,2,0]) # Order axes naturally

    def C6_scatter(self, S, V, surface, rho):
        """
        Attempting to mix PARALLEL control variates with BS evaluation.
        We follow Bergomi (2015) but
        fix the log S QV budget at the known maximum so that the control doesn't
        have to change (incurring jumps at unknown random stopping times). We
        choose an optimum scaling factor to reduce variance. Here the payoff is
        computed and the BS formula used on full paths. Next we will try
        combining a control variant with C3 parallel BS computations. The
        resulting cube takes dimensions N x M x K = paths x expiries x strikes.
        This is required to obtain error bounds on the implied call prices.
        """
        K = np.exp(np.array(surface._log_strike_surface())) # Assumes S0 = 1
        M = (surface._maturities * self.n).astype(int) # Location of expiries
        T = surface._maturities

        # Construct integrated variances for conditional vols
        # More efficient to use variances directly in BS rather than vols
        sigma = np.sqrt(np.cumsum(V, axis = 1) * self.dt / self.t)

        # Compute maximum QV of log S achieved
        # This should be at EVERY desired time slice!
        maxQV = np.amax(sigma[:,-1])**2 * T[-1]

        C = np.zeros((len(V[:,0]), len(K[:,0]), len(K[0,:])))
        D = np.zeros((len(V[:,0]), len(K[:,0]), len(K[0,:]))) # Control
        eCV = np.zeros((len(D[0,:,0]), len(D[0,0,:])))
        for i in range(len(M[:,0])):
            # Construction of sigma is sexy
            # Collapses to payoff in rho = +-1 case
            # So evaluation of BS completely redundant
            sigmaT = np.sqrt((1 - rho**2)) * sigma[:,M[i,0]][:,np.newaxis]
            QVT = sigma[:,M[i,0]][:,np.newaxis]**2 * T[i,0]
            maxQVT = np.max(QVT)

            ST = S[:,M[i,0]][:,np.newaxis]
            KT = K[i,:][np.newaxis,:]
            # C[:,i,:] = np.maximum(ST - KT, 0) # Note broadcasting
            # Must pass only parallel ST into this method!!
            C[:,i,:] = bs(ST, KT, sigmaT**2 * T[i,0]) # Note broadcasting
            # C[:,i,:] = np.maximum(ST - KT, 0) # Note broadcasting

            if rho != 0.:
                D[:,i,:] = bs(ST, KT, rho**2*(maxQVT - QVT)) # Control

                # Just need to broadcast this properly
                # Expectation of control
                eCV[i,:] = bs(1.0, KT, rho**2 * maxQVT) # Broadcast to K shape

        # E[D] did not give eCV when I tried using only parallel part in
        # control -- have to think about this...

        # Now must compute optimal control constant
        # Must be careful with broadcasting for strikes
        # Assumes single maturity
        if rho != 0.:
            c = np.zeros((1,len(K[:,0]),len(K[0,:])))
            for j in range(len(K[:,0])):
                for i in range(len(K[0,:])):
                    covMat = np.cov(C[:,j,i], D[:,j,i])
                    c[0,j,i] = - covMat[0,1] / covMat[1,1]

        # Expectation of control
        # eCV = bs(1.0, KT, rho**2 * maxQVT) # Broadcast to K shape
        # eCV = np.zeros((len(D[0,:,0]), len(D[0,0,:])))
        # for i in range(len(M[:,0])):
        #     eCV[i,:] = np.mean(D[:,i,:], axis = 0) # Broadcast to K shape
        #
        # # Just need to broadcast this properly
        # # Expectation of control
        # eCV = bs(1.0, KT, rho**2 * maxQVT) # Broadcast to K shape

        # Finally make new sample
        if rho != 0.:
            E = C + c*(D - eCV)
        else:
            E = C

        # Should instead return block and report upon it
        # Return only expectation and standard deviation
        e = np.mean(E, axis = 0)
        s = np.std(E, axis = 0) / np.sqrt(len(E[:,0,0])) # Division by sqrt paths
        return np.array([C,D]) # Order axes naturally

    def C7(self, S, V, surface):
        """
        JUST IMPLEMENT SINGLE TIMER CONTROL FIRST FOR ARBITRARY BUDGET, THEN
        ADD ASSET INVESTMENT TO ASSESS IMPROVEMENT

        Attempting to fully replicate Bergomi's solution on p 341.
        Presently trying to salvage some elegance by considering this as a
        multi-control problem. I think this works, elegantly.
        The multi-controls could be set e.g. at the expected QV level and 1SD
        above and below (in log space).
        See p 141 of Stocastic Simulation (Asmussen, Glynn 2007)
        """
        K = np.exp(np.array(surface._log_strike_surface())) # Assumes S0 = 1
        M = (surface._maturities * self.n).astype(int) # Location of expiries
        T = surface._maturities

        # Construct integrated variances for conditional vols
        # More efficient to use variances directly in BS rather than vols
        sigma = np.sqrt(np.cumsum(V, axis = 1) * self.dt / self.t)
        QV = np.cumsum(V, axis = 1) * self.dt

        # Compute maximum QV of log S achieved
        # This should be at EVERY desired time slice!
        # Clearly this should be seperate method !!!
        maxQV = np.amax(QV[:,-1])
        avQV = np.mean(QV[:,-1])

        budget = 2 * avQV # SET CONTROL VARIATE HERE
        # SHOULD BE MULTIPLE OF ANALYTIC AVERAGE SINCE POINT IS TO AVOID
        # INFLUENCE OF OUTLIERS

        bools = QV < budget
        # try not removing 1 and check fails
        tau = np.sum(bools, axis = 1) - 1 # finds last, not exceeding

        S_tau = S[range(len(tau)), tau][:,np.newaxis] # compiles stopped entries
        QV_tau = QV[range(len(tau)), tau][:,np.newaxis] # compiles stopped entries

        C = np.zeros((len(V[:,0]), len(K[:,0]), len(K[0,:])))
        D = np.zeros((len(V[:,0]), len(K[:,0]), len(K[0,:]))) # Control
        for i in range(len(M[:,0])):
            # Construction of sigma is sexy
            # Collapses to payoff in rho = +-1 case
            # So evaluation of BS completely redundant
            # sigmaT = np.sqrt((1 - rho**2)) * sigma[:,M[i,0]][:,np.newaxis]
            QVT =  sigma[:,M[i,0]][:,np.newaxis]**2 * T[-1]

            ST = S[:,M[i,0]][:,np.newaxis]
            KT = K[i,:][np.newaxis,:]
            # C[:,i,:] = np.maximum(ST - KT, 0) # Note broadcasting
            # C[:,i,:] = blsprice(ST, KT, T, sigmaT) # Note broadcasting
            C[:,i,:] = np.maximum(ST - KT, 0) # Note broadcasting
            D[:,i,:] = bs(S_tau, KT, budget - QV_tau) # * ST/S_tau # Control
            # SET REINVESTMENT AS OPTION

        # Now must compute optimal control constant
        # Must be careful with broadcasting for strikes
        # Assumes single maturity
        c = np.zeros((1,1,len(K[0,:])))
        for i in range(len(K[0,:])):
            covMat = np.cov(C[:,0,i], D[:,0,i])
            c[0,0,i] = - covMat[0,1] / covMat[1,1]

        # Expectation of control
        eCV = bs(1.0, KT, budget) # Broadcast to K shape

        # Finally make new sample
        E = C + c * (D - eCV)

        # Should instead return block and report upon it
        # Return only expectation and standard deviation
        e = np.mean(E, axis = 0)
        s = np.std(E, axis = 0) / np.sqrt(len(E[:,0,0])) # Division by sqrt paths
        return np.transpose([e,s], [1,2,0]) # Order axes naturally

    def C8(self, S, V, surface, rho):
        """
        parallel part with simple S hedge
        """
        K = np.exp(np.array(surface._log_strike_surface())) # Assumes S0 = 1
        M = (surface._maturities * self.n).astype(int) # Location of expiries
        T = surface._maturities

        # Construct integrated variances for conditional vols
        # More efficient to use variances directly in BS rather than vols
        sigma = np.sqrt(np.cumsum(V, axis = 1) * self.dt / self.t)

        # Compute maximum QV of log S achieved
        # This should be at EVERY desired time slice!
        # maxQV = np.amax(sigma[:,-1])**2 * T[-1]

        C = np.zeros((len(V[:,0]), len(K[:,0]), len(K[0,:])))
        D = np.zeros((len(V[:,0]), len(K[:,0]), len(K[0,:]))) # Control
        for i in range(len(M[:,0])):
            # Construction of sigma is sexy
            # Collapses to payoff in rho = +-1 case
            # So evaluation of BS completely redundant
            sigmaT = np.sqrt((1 - rho**2)) * sigma[:,M[i,0]][:,np.newaxis]
            # QVT = sigma[:,M[i,0]][:,np.newaxis]**2 * T[-1]

            ST = S[:,M[i,0]][:,np.newaxis]
            KT = K[i,:][np.newaxis,:]
            # C[:,i,:] = np.maximum(ST - KT, 0) # Note broadcasting
            # Must pass only parallel ST into this method!!
            C[:,i,:] = bs(ST, KT, sigmaT**2 * T) # Note broadcasting
            # C[:,i,:] = np.maximum(ST - KT, 0) # Note broadcasting
            D[:,i,:] = ST # blsprice(ST, KT, rho**2*(maxQV - QVT), 1.0) # Control

        # E[D] did not give eCV when I tried using only parallel part in
        # control -- have to think about this...

        # Now must compute optimal control constant
        # Must be careful with broadcasting for strikes
        # Assumes single maturity
        c = np.zeros((1,1,len(K[0,:])))
        for i in range(len(K[0,:])):
            covMat = np.cov(C[:,0,i], D[:,0,i])
            c[0,0,i] = - covMat[0,1] / covMat[1,1]

        # Expectation of control
        eCV = 1. # blsprice(1.0, KT, rho**2 * maxQV, 1.0) # Broadcast to K shape

        # Finally make new sample
        E = C + c * (D - eCV)

        # Should instead return block and report upon it
        # Return only expectation and standard deviation
        e = np.mean(E, axis = 0)
        s = np.std(E, axis = 0) / np.sqrt(len(E[:,0,0])) # Division by sqrt paths
        return np.transpose([e,s], [1,2,0]) # Order axes naturally

    def C9(self, S, surface):
        """
        Returns call option payoffs from a price process and surface. The
        resulting cube takes dimensions N x M x K = paths x expiries x strikes
        This is required to obtain error bounds on the implied call prices
        """
        K = np.exp(np.array(surface._log_strike_surface())) # Assumes S0 = 1
        M = (surface._maturities * self.n).astype(int) # Location of expiries

        C = np.zeros((len(S[:,0]), len(K[:,0]), len(K[0,:])))
        D = np.zeros((len(S[:,0]), len(K[:,0]), len(K[0,:]))) # Control
        for i in range(len(M[:,0])):
            ST = S[:,M[i,0]][:,np.newaxis]
            KT = K[i,:][np.newaxis,:]
            C[:,i,:] = np.maximum(ST - KT, 0) # Note broadcasting
            D[:,i,:] = ST # blsprice(ST, KT, rho**2*(maxQV - QVT), 1.0) # Control

        c = np.zeros((1,1,len(K[0,:])))
        for i in range(len(K[0,:])):
            covMat = np.cov(C[:,0,i], D[:,0,i])
            c[0,0,i] = - covMat[0,1] / covMat[1,1]

        eCV = 1. # blsprice(1.0, KT, rho**2 * maxQV, 1.0) # Broadcast to K shape

        # Finally make new sample
        E = C + c * (D - eCV)

        # Return only expectation and standard deviation
        e = np.mean(E, axis = 0)
        s = np.std(E, axis = 0) / np.sqrt(len(E[:,0,0])) # Division by sqrt paths
        return np.transpose([e,s], [1,2,0]) # Order axes naturally

    def IV(self, C, surface):
        """
        Returns implied Black volatility from call option prices.
        """
        # Vectorise implied vol function using NumPy (uses loops)
        # Rename this crap
        vblsimpv = np.vectorize(blsimpv)

        K = np.exp(np.array(surface._log_strike_surface())) # Assumes S0 = 1
        T = surface._maturities # Maturities

        # Construct implied vols
        IV = np.zeros((len(K[:,0]), len(K[0,:])))
        IV = vblsimpv(C[:,:,0], 1., K, T) # Expectation
        return IV

    def IVP(self, P, surface):
        """
        Same as above but for puts.
        """
        # Vectorise implied vol function using NumPy (uses loops)
        # Rename this crap
        vblsimpv = np.vectorize(blsimpvp)

        K = np.exp(np.array(surface._log_strike_surface())) # Assumes S0 = 1
        T = surface._maturities # Maturities

        # Construct implied vols
        IV = np.zeros((len(K[:,0]), len(K[0,:])))
        IV = vblsimpv(P[:,:,0], 1., K, T) # Expectation
        return IV

    def IV_from_k(self, C, log_strikes):
        """
        Returns implied Black volatility from call option prices. This is
        specifically for the generation of a surface over a fixed range of
        log-strikes and for a fixed maturity.
        """
        # Vectorise implied vol function using NumPy (uses loops)
        # Rename this crap
        vblsimpv = np.vectorize(blsimpv)

        K = np.exp(log_strikes) # Assumes S0 = 1
        T = self.T # Maturities

        # Construct implied vols
        IV = np.zeros((len(K[:,0]), len(K[0,:])))
        IV = vblsimpv(C[:,:,0], 1., K, T) # Expectation
        return IV

    def C_from_k(self, S, log_strikes):
        """
        Returns call option payoffs from a price process and surface. The
        resulting cube takes dimensions N x M x K = paths x expiries x strikes
        This is required to obtain error bounds on the implied call prices
        """
        K = np.exp(log_strikes) # Assumes S0 = 1
        M = np.array([[312]]) # Location of expiries

        C = np.zeros((len(S[:,0]), len(K[:,0]), len(K[0,:])))
        for i in range(len(M[:,0])):
            ST = S[:,M[i,0]][:,np.newaxis]
            KT = K[i,:][np.newaxis,:]
            C[:,i,:] = np.maximum(ST - KT, 0) # Note broadcasting

        # Return only expectation and standard deviation
        e = np.mean(C, axis = 0)
        s = np.std(C, axis = 0) / np.sqrt(len(C[:,0,0])) # Division by sqrt paths
        return np.transpose([e,s], [1,2,0]) # Order axes naturally

    # This is old method which also gives confidence bounds
    # def IV(self, C, surface):
    #     """
    #     Returns implied Black volatility from call option prices and their 1sd
    #     Can't have this returning 1SDs if later to be used in solver.
    #     """
    #     # Vectorise implied vol function using NumPy (uses loops)
    #     # Rename this crap
    #     vblsimpv = np.vectorize(blsimpv)
    #
    #     K = np.exp(np.array(surface._log_strike_surface())) # Assumes S0 = 1
    #     T = surface._maturities # Maturities
    #
    #     # Construct implied vols + 1SD error bounds
    #     IV = np.zeros((len(K[:,0]), len(K[0,:]), 3))
    #     IV[:,:,0] = vblsimpv(C[:,:,0] - 1 * C[:,:,1], 1., K, T) # - 1SD
    #     IV[:,:,1] = vblsimpv(C[:,:,0]               , 1., K, T) # Expectation
    #     IV[:,:,2] = vblsimpv(C[:,:,0] + 1 * C[:,:,1], 1., K, T) # + 1SD
    #
    #     return IV
