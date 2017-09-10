import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from surface_utils import *

'''
TODO:
x Missing ability to construct skew over time. This is important because the
  roughness index is linked to this curve. Should be construct for each
  maturity slice using the derivative at 0 of a spline of IV w.r.t k
'''

class Surface(object):
    """

    """
    def __init__(self, index, date, scale = 100, close = False):
        """
        Constructor for useful private data and public Pandas surface.
        """
        # Import data
        data = xl_import(index, date, close = close)

        # Basic data assignments; all private
        self._raw_data = data
        self._date = int(float(data[0,1]))
        self._currency = data[1,1]
        self._spot = float(data[2,1])

        # Flip and transpose surface for use
        self._raw_surface = np.transpose(np.flipud(data[5:,1:])).astype(float)

        # 0D string arrays for Pandas keys
        self._tenors = data[4,1:]
        self._deltas = np.flipud(data[5:,0]) # Flip to obtain puts first

        # 1D numerical arrays for NumPy broadcasting
        self._forwards = data[3,1:].astype(float)[:,np.newaxis]
        self._maturities = tenors_to_yearfracs(self._tenors)[:,np.newaxis]

        # Build Pandas dataframe of surface; transpose of raw data
        self.surface = pd.DataFrame(self._raw_surface/scale,
                                    index = self._tenors,
                                    columns = self._deltas)

    # Should write plot_all method once happy with each
    # These should be deprecated. Plots should take place in IPython notebook
    def plot_forwards(self):
        """
        Basic plot of forwards against maturities.
        """
        date = str(self._date)
        currency = self._currency

        title = currency + ' Forwards as of ' + date # Should be string naturally
        xlabel = 'Maturity (Years)'
        ylabel = 'Forward'

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        xvals = np.append(0, self._maturities)
        yvals = np.append(self._spot, self._forwards)
        plt.plot(xvals, yvals)

        # Need to use yaml config file for this
        install_path = '/Users/ryanmccrickerd/desktop'
        save_path = install_path + '/rbergomi/plots/' + date + '_' + currency + '_forwards.png'

        # Save file
        plt.savefig(save_path)

    def _put_deltas(self):
        """
        Returns the put delta, N(-d1), of each point on the volatility surface.
        The put delta is more natural than the call, because it orientates the
        surface to imply increasing strikes along the x-axis.
        """
        # Prepare delta_to_put function for NumPy array operation
        deltas_to_puts = np.vectorize(delta_to_put)

        # Assign arguments
        deltas = self._deltas[np.newaxis,:]
        maturities = self._maturities
        volatilities = np.array(self.surface)

        # Construct delta surface
        put_deltas = deltas_to_puts(deltas, maturities, volatilities)

        # Return as Pandas surface
        return pd.DataFrame(put_deltas, index = self._tenors,
                            columns = self._deltas)

    def _strike_surface(self):
        """
        Returns the surface of strikes implied by the volatility surface.
        """
        # Assign everything needed for put_delta_to_strike function
        forwards = self._forwards
        put_deltas = np.array(self._put_deltas())
        volatilities = np.array(self.surface)
        maturities = self._maturities

        # Construct surface
        strike_surface = put_delta_to_strike(forwards, put_deltas, volatilities,
                                             maturities)
        # Return as Pandas surface
        return pd.DataFrame(strike_surface, index = self._tenors,
                            columns = self._deltas)

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

    # Deprecate
    def plot_surface(self, x_axis = 'Delta'):
        """
        Save plot of surface against put deltas, strikes or log-strikes.
        """
        date = str(self._date)
        currency = self._currency

        title = currency + ' Surface as of ' + date
        xlabel = x_axis
        ylabel = 'Volatility'

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Add plot each tenor in the surface
        for tenor in self._tenors:
            # Assign xvals based on x_axis
            if x_axis == 'Strike':
                xvals = np.array(self._strike_surface().loc[tenor])

            elif x_axis == 'Log-Strike':
                xvals = np.array(self._log_strike_surface().loc[tenor])

            else:  xvals = np.array(self._put_deltas().loc[tenor])

            yvals = np.array(self.surface.loc[tenor])
            plt.plot(xvals, yvals)

        # Need to use yaml config file for this
        install_path = '/Users/ryanmccrickerd/desktop'
        save_path = install_path + '/rbergomi/plots/' + date + '_' + currency + '_surface.png'

        # Add legend and save file
        plt.legend(self._tenors)
        plt.savefig(save_path)

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

    # Deprecate
    def plot_variance(self, degree = 3, smoothing = 0, discretisation = 1e-3):
        """
        Save plot of variance and instantaneous forward variance.
        """
        date = str(self._date)
        currency = self._currency

        title = currency + ' Variance as of ' + date
        xlabel = 'Maturity (Years)'
        ylabel = 'Variance'

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Assign curve to be plotted, and derivative
        variance_curve = self._variance_curve(degree, smoothing)
        deriv_curve = variance_curve.derivative()

        # Make discretised time grid
        tvals = np.arange(0, self._maturities[-1,0], discretisation)

        # Plot integrated and instantaneous forward variances
        integrated = variance_curve(tvals)
        plt.plot(tvals, integrated)
        # This is via differention of definition
        forward = integrated + tvals * deriv_curve(tvals)
        plt.plot(tvals, forward)

        # Need to use yaml config file for this
        install_path = '/Users/ryanmccrickerd/desktop'
        save_path = install_path + '/rbergomi/' + date + '_' + currency + '_variance.png'

        # Add legend and save file
        plt.legend(['Integrated','Instantaneous Forward'])
        plt.savefig(save_path)

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

    def g(self, beta = 1.0): # Choosing rBergomi by default
        """
        Returns the second derivative of a payoff function, g''(K), evaluated
        on a surface, as required in Gatheral (11.1). This is specifically for
        the CEV case, g(x) = x^(2b - 2), g''(x) = (2b - 2)(2b - 3) x^(2b - 4).
        """
        K = np.array(self._strike_surface())/self._forwards
        g = (2*beta - 2) * (2*beta - 3) * K**(2*beta - 4)
        return g

    def integrands(self, beta = 1.0):
        """
        Returns the put and call integrands in Gatheral (11.1).
        Should the boundary conditions be imposed here?
        """
        g = self.g(beta = beta)
        P = self.prices(option = 'Put')
        C = self.prices(option = 'Call')
        return [P * g, C * g]

    def integrand_splines(self, beta = 1.0):
        """
        Splines for Gatheral (11.1) integrands, preparing for integration.
        Must extend these through (0,0) and (10,0) and ensure non-negative.
        Make beta an attribute of the class, when tidying up.
        """
        K = np.array(self._strike_surface())/self._forwards
        # Get integrands for puts and calls, including g'' weightings
        P, C = self.integrands(beta = beta)

        # Impose boundaries on put (0,0) and call (10,0) surfaces
        nbK = len(K[0,:]) # Number of strikes
        KP = np.insert(K,   0, 0, axis = 1) # Enter K = 0 at beginning
        KC = np.insert(K, nbK, 2, axis = 1) # Enter K = 2 at end
        P  = np.insert(P,   0, 0, axis = 1) # Enter P = 0 at beginning
        C  = np.insert(C, nbK, 0, axis = 1) # Enter C = 0 at end

        spl = UnivariateSpline # Just to save space

        # Empty lists for splines
        P_spl = len(K[:,0]) * [None]
        C_spl = len(K[:,0]) * [None]

        # Construct splines for each maturity
        for i in range(len(K[:,0])):
            P_spl[i] = spl(KP[i,:], P[i,:], k = 2, s = 0) # bbox should be natural
            C_spl[i] = spl(KC[i,:], C[i,:], k = 2, s = 0)

        return [P_spl, C_spl]
