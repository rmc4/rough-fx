import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from rbergomi.surface_utils import *

class Surface(object):
    """

    """
    def __init__(self, currency, date, scale=100):
        """
        Constructor for surface instance.
        """
        df = csv_import(currency, date)
        self.ccy = currency
        self.date = date
        self.surface = df/scale
        self.deltas = np.array(df.columns).astype(str)
        self.tenors = np.array(df.index).astype(str)
        self.maturities = tenors_yearfracs(self.tenors)
        self.vols = np.array(self.surface)

    def put_deltas(self):
        """
        Returns the put delta, N(-d1), of each point on the volatility surface.
        The put delta is more natural than the call, because it orientates the
        surface to imply increasing strikes along the x-axis.
        """
        # Prepare delta_to_put function for NumPy array operation
        deltas_to_puts = np.vectorize(delta_to_put)

        # Assign arguments
        deltas = self.deltas[np.newaxis,:]
        maturities = self.maturities[:,np.newaxis]
        vols = self.vols

        # Construct delta surface
        put_deltas = deltas_to_puts(deltas, maturities, vols)

        # Return as Pandas surface
        return pd.DataFrame(put_deltas, index=self.tenors, columns=self.deltas)

    def strikes(self):
        """
        Returns the surface of strikes implied by the volatility surface.
        """
        # Assign everything needed for put_delta_to_strike function
        put_deltas = np.array(self.put_deltas())
        vols = self.vols
        maturities = self.maturities[:,np.newaxis]

        # Construct surface
        strikes = put_delta_to_strike(put_deltas, vols, maturities)

        # Return as Pandas surface
        return pd.DataFrame(strikes, index=self.tenors, columns=self.deltas)

    def logstrikes(self):
        """
        Transforms a strike surface into log-strikes, for given forwards.
        """
        # Assign everything needed for strike_to_log_strike function
        strikes = np.array(self.strikes())

        logstrikes = strike_to_log_strike(strikes)

        # Return as Pandas surface
        return pd.DataFrame(logstrikes, index=self.tenors, columns=self.deltas)

    def forward_deltas(self):
        """
        Converts surface of put deltas, N(-d1), to surface of forward deltas,
        N(-d2), required in the variance integral, Austing (10.28).
        """
        # Assign arguments for put_to_forward
        put_deltas = np.array(self.put_deltas())
        maturities = self.maturities[:,np.newaxis]
        vols = self.vols

        # Construct forward delta, N(-d2), surface
        forward_deltas = put_to_forward(put_deltas, maturities, vols)

        # Return as Pandas surface
        return pd.DataFrame(forward_deltas, index=self.tenors, columns=self.deltas)

    def variance_splines(self, degree = 3, smoothing = 0):
        """
        Constructs a spline for each tenor-slice of the squared volatility
        surface, which will be integrated w.r.t. forward delta, Austing (10.28).
        """
        # Assign arguments
        forward_deltas = self.forward_deltas()
        vols = self.surface

        variance_splines = len(self.tenors) * [None] # Empty list
        i = 0
        for tenor in self.tenors:
            deltas = np.array(forward_deltas.loc[tenor])
            variances = np.array(vols.loc[tenor])**2
            spline = UnivariateSpline(deltas, variances, bbox=[0,1], k=degree,
                                      s=smoothing)
            variance_splines[i] = spline
            i += 1

        # Return spline for each tenor as Python list
        return variance_splines

    def skew(self, degree = 3, smoothing = 0):
        """
        Constructs a spline for each tenor-slice of the volatility
        surface against log-strike, k, to obtain ATM skew (Bayer 15 p. 4).
        """
        # Assign arguments
        k = self.logstrikes()
        surface = self.surface

        skew = len(self.tenors) * [None] # Empty list
        i = 0
        for tenor in self.tenors:
            ks = np.array(k.loc[tenor])
            vols = np.array(surface.loc[tenor])
            spline = UnivariateSpline(ks, vols, k=degree,
                                      s=smoothing)

            deriv = spline.derivative()
            skew[i] = deriv(0)
            i += 1

        # Return spline for each tenor as Python list
        return np.array(skew)

    def integrated_variances(self, degree = 3, smoothing = 0):
        """
        Integrates variance splines, Austing (10.28), to obtain integrated
        variances for each tenor.
        """
        # Assign splines to be integrated
        variance_splines = self.variance_splines(degree, smoothing)

        integrated_variances = np.zeros(len(self.tenors))
        i = 0
        for tenor in self.tenors:
            # Integrate each spline in (0,1)
            integrated_variances[i] = variance_splines[i].integral(0,1)
            i += 1

        # Return as Pandas DataFrame
        return pd.DataFrame(integrated_variances, index = self.tenors)

    def variance_curve(self, degree = 3, smoothing = 0):
        '''
        Contructs an integrated variance curve from the discrete points of
        _integrated_variances which can be evaluated at any time. Particularly
        important for evaluating discretised or instantaneous forward variance.
        '''
        # Assign useful data
        maturities = self.maturities
        integrated_variances = np.array(self.integrated_variances(degree, smoothing)) # Need option of piecewise constant forward variance (analytic)

        # Polynomial degree given by k; s=0 ensures through every point
        spline = UnivariateSpline(maturities, integrated_variances,
                                  bbox=[0,maturities[-1]], k=degree,
                                  s=smoothing)
        return spline

    def ξ(self, n=156):
        """
        Construct piecewise flat forward variance curve, xi, analytically.
        Returns discretised grid which can be broadcast against a variance
        process.
        """

        M = self.maturities
        T = M[-1] # Max maturity
        s = 1 + int(n * T) # Length of array
        ξ = np.zeros((1, s))
        t = np.linspace(0, T, 1 + s) # Time grid

        # This integrates with cubic spline, loosing accuracy
        # V = np.array(self.integrated_variances())

        # Testing improvment with SVI
        V = (np.array(self.SVI().loc[:,'V'])/self.maturities)[:,np.newaxis]

        Vf = np.zeros_like(V) # Forward variance
        Vf[0] = V[0]
        for i in range(len(M) - 1): # Strip forwards
            Vf[i+1] = (M[i+1] * V[i+1] - M[i] * V[i]) / (M[i+1] - M[i])

        j = 0
        for i in range(s):
            if t[i] >= M[j]:
                j += 1
            ξ[0,i] = Vf[j]

        return ξ

    def prices(self, option = 'Call'):
        """
        Produce surface of forward call prices. Scalings such that forwards = 1
        are assumed (working with normalised discounted process).
        """
        # Assign scaled requirements
        F = np.ones_like(self._forwards)
        K = np.array(self._strike_surface())/self._forwards
        T = self._maturities
        vol = np.array(self.surface)

        # Broadcast B-S call or put prices
        if option == 'Call':
            prices = call_price(F, K, T, vol)
        else:
            prices = put_price(F, K, T, vol)
        return prices

    def SVI(self):
        """
        Using notation from VIX futures (JMM17) p.15
        σ^2_BS(k)t = w = θ/2 * (1 + ρ*φ*k + np.sqrt((φ*k + ρ)**2 + 1 - ρ**2))
        Relies on there being an ATM column
        """
        # Prepare basic things
        σ = self.vols
        ATM = np.array(self.surface.loc[:,'ATM'])
        t = self.maturities
        θ = ATM**2*t
        k = np.array(self.logstrikes())

        headers = ['t','Θ','ρ','φ','RMSE','arb','V','d0','d1','d2']
        array = np.zeros((len(t),len(headers)))

        # Prepare solver params
        ρ0,φ0 = -0.5, 50
        ρ1,φ1 = (-1,1),(-100,100)
        x0, bnds = (ρ0,φ0),(ρ1,φ1)
        ops = {'maxiter':100}

        for i in range(len(t)):
            # Define objective function for time slice and minimise
            def rmse(x):
                ρ,φ = x[0],x[1]
                true = σ[i,:]
                aprx = SVI(k[i,:],t[i],θ[i],ρ,φ)
                rmse = np.sqrt(np.mean((aprx-true)**2))
                return rmse
            res = minimize(rmse,x0,method='L-BFGS-B',bounds=bnds,options=ops)

            # Build slice for dataframe output
            ρ = res.x[0]
            φ = res.x[1]
            χ = (1-ρ**2)*θ[i]*φ/4
            a = 1+θ[i]*φ/2*(ρ-χ/2)
            b = θ[i]*φ*(χ-ρ)
            c = θ[i]*φ*χ
            RMSE = res.fun
            arb =  ( θ[i]*φ**2 * (1 + np.abs(ρ)) ) > 4
            V = (b**2+2*a*(c+θ[i]))/2/a**2
            d0 = np.sqrt(θ[i]/t[i])
            d1 = ρ*φ*np.sqrt(θ[i]/t[i])/2
            d2 = φ**2*np.sqrt(θ[i]/t[i])*(-2*ρ**2 + 1)/4
            array[i,:] = [ t[i],θ[i],ρ,φ,RMSE,arb,V,d0,d1,d2 ]

        return pd.DataFrame(array, index=self.tenors, columns=headers)
