import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from rbergomi.surface_utils import *

class Surface(object):
    """

    """
    def __init__(self, currency, date, scale=100):
        """
        Constructor for surface instance.
        """
        df = csv_import(currency, date)
        self.surface = df
        self.deltas = np.array(df.columns).astype(str)
        self.tenors = np.array(df.index).astype(str)
        self.maturities = tenors_yearfracs(self.tenors)
        self.vols = np.array(df)/scale

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

    def strike_surface(self):
        """
        Returns the surface of strikes implied by the volatility surface.
        """
        # Assign everything needed for put_delta_to_strike function
        put_deltas = np.array(self.put_deltas())
        vols = self.vols
        maturities = self.maturities

        # Construct surface
        strikes = put_delta_to_strike(put_deltas, vols, maturities)

        # Return as Pandas surface
        return pd.DataFrame(strikes, index=self.tenors, columns=self.deltas)

    def _log_strike_surface(self):
        """
        Transforms a strike surface into log-strikes, for given forwards.
        """
        # Assign everything needed for strike_to_log_strike function
        forwards = self._forwards
        strikes = np.array(self._strike_surface())

        log_strike_surface = strike_to_log_strike(forwards, strikes)

        # Return as Pandas surface
        return pd.DataFrame(log_strike_surface, index = self._tenors,
                            columns = self._deltas)

    def _forward_deltas(self):
        """
        Converts surface of put deltas, N(-d1), to surface of forward deltas,
        N(-d2), required in the variance integral, Austing (10.28).
        """
        # Assign arguments for put_to_forward
        put_deltas = np.array(self._put_deltas())
        maturities = self._maturities
        volatilities = np.array(self.surface)

        # Construct forward delta, N(-d2), surface
        forward_deltas = put_to_forward(put_deltas, maturities, volatilities)

        # Return as Pandas surface
        return pd.DataFrame(forward_deltas, index = self._tenors,
                            columns = self._deltas)

    def _variance_splines(self, degree = 3, smoothing = 0):
        """
        Constructs a spline for each tenor-slice of the squared volatility
        surface, which will be integrated w.r.t. forward delta, Austing (10.28).
        """
        # Assign arguments
        forward_deltas = self._forward_deltas()
        surface = self.surface

        variance_splines = len(self._tenors) * [None] # Empty list
        i = 0
        for tenor in self._tenors:
            deltas = np.array(forward_deltas.loc[tenor])
            variances = np.array(surface.loc[tenor])**2
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
        k = self._log_strike_surface()
        surface = self.surface

        skew = len(self._tenors) * [None] # Empty list
        i = 0
        for tenor in self._tenors:
            ks = np.array(k.loc[tenor])
            vols = np.array(surface.loc[tenor])
            spline = UnivariateSpline(ks, vols, k=degree,
                                      s=smoothing)

            deriv = spline.derivative()
            skew[i] = np.absolute(deriv(0))
            i += 1

        # Return spline for each tenor as Python list
        return np.array(skew)

    def _integrated_variances(self, degree = 3, smoothing = 0):
        """
        Integrates variance splines, Austing (10.28), to obtain integrated
        variances for each tenor.
        """
        # Assign splines to be integrated
        variance_splines = self._variance_splines(degree, smoothing)

        integrated_variances = np.zeros(len(self._tenors))
        i = 0
        for tenor in self._tenors:
            # Integrate each spline in (0,1)
            integrated_variances[i] = variance_splines[i].integral(0,1)
            i += 1

        # Return as Pandas DataFrame
        return pd.DataFrame(integrated_variances, index = self._tenors)

    def _variance_curve(self, degree = 3, smoothing = 0):
        '''
        Contructs an integrated variance curve from the discrete points of
        _integrated_variances which can be evaluated at any time. Particularly
        important for evaluating discretised or instantaneous forward variance.
        '''
        # Assign useful data
        maturities = self._maturities[:,0]
        integrated_variances = np.array(self._integrated_variances(degree, smoothing)) # Need option of piecewise constant forward variance (analytic)

        # Polynomial degree given by k; s=0 ensures through every point
        spline = UnivariateSpline(maturities, integrated_variances,
                                  bbox=[0,maturities[-1]], k=degree,
                                  s=smoothing)
        return spline

    def xi(self, n = 2 * 156):
        """
        Construct piecewise flat forward variance curve, xi, analytically.
        Returns discretised grid which can be broadcast against a variance
        process.
        """

        M = self._maturities
        T = M[-1,0] # Max maturity
        s = 1 + int(n * T) # Length of array
        xi = np.zeros((1, s))
        t = np.linspace(0, T, 1 + s) # Time grid


        V = np.array(self._integrated_variances())

        Vf = np.zeros_like(V) # Forward variance
        Vf[0] = V[0]
        for i in range(len(M[:,0]) - 1): # Strip forwards
            Vf[i+1] = (M[i+1] * V[i+1] - M[i] * V[i]) / (M[i+1] - M[i])

        j = 0
        for i in range(s):
            if t[i] >= M[j]:
                j += 1
            xi[0,i] = Vf[j]

        return xi

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
