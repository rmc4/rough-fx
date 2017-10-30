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
    def __init__(self, n=256, N=1024, T=1.0):
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


        # fn = 'sobol/'+str(seed)+'-'+str(self.N)+'-'+str(4*s)+'.csv'
        # random_numbers = np.array(pd.read_csv(fn))

        ## SHOULD BE OUTSIDE CALIBRATION ROUTINE
        np.random.seed(seed)
        random_numbers = np.random.normal(size=(self.N,4*s))

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
        # for i in np.arange(1, 1 + self.s, 1): # See (3.6)
        #     Y1[:,i] += dW[:,i-1,1] # Assumes kappa = 1

        # Construct Y1 through exact integral
        Y1[:,1:1+self.s] = dW[:,:self.s,1] # Assumes kappa = 1

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
        return Y

    # Yes should raise dimens
    def V(self, Yα, Yβ, ξ=1.0, ζ=-0.5, η=1.5):
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

    def surface(self, S, surf):
        """
        Provides the implied Black volatility surface for every option
        implicitely in a Surface object.
        """
        vec_bsinv = np.vectorize(bsinv)

        indices = (surf.maturities * self.n).astype(int)
        ST = S[:,indices][:,:,np.newaxis]
        K = np.array(surf.strikes())[np.newaxis,:,:]
        Δ = np.array(surf.forward_deltas())
        T = surf.maturities[:,np.newaxis]

        call_payoffs = np.maximum(ST - K,0) - (1-Δ)*(ST - 1)
        call_prices = np.mean(call_payoffs, axis=0)
        call_vols = vec_bsinv(call_prices, 1., np.squeeze(K), T, ϕ=1)

        put_payoffs = np.maximum(K - ST,0) + Δ*(ST - 1)
        put_prices = np.mean(put_payoffs, axis=0)
        put_vols = vec_bsinv(put_prices, 1., np.squeeze(K), T, ϕ=-1)

        # don't think helpful when have control
        vols = (call_vols + put_vols) / 2

        return pd.DataFrame(vols, index=surf.tenors, columns=surf.deltas)
