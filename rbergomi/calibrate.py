import numpy as np
from scipy.optimize import minimize
from datetime import datetime # For calibration timer

class Calibrate(object):
    """
    Class for calibrating a specified rBergomi model to a Surface object.
    After running the passed rBergomi instance must hold calibrated results.
    Need to think about extending this more naturally for other TBSS processes
    given than Y and v need amending in each case.
    """
    def __init__(self, rbergomi, surface, gamma = False, b = 0.0001, seed = 0):
        # Assign rBergomi and Surface instances to Calibrate
        self.rbergomi = rbergomi # rBergomi instance should inherit all results
        self.surface = surface # Only k and xi methods required

        # Extract helpful results from Surface instance
        self.k = np.array(surface._log_strike_surface()) # Tidy up
        self.M = surface._maturities
        self.xi = surface.xi(n = self.rbergomi.n) # Awful
        self.actual_surface = np.array(surface.surface)

        # Store scipy.optimize.minimize requirements
        self.seed = seed # Not actually consumed by minimize

        # Set seed here before the single call for increments
        np.random.seed(self.seed)
        self.dW1 = self.rbergomi.dW1() # Runs fast if already assigned?
        self.dW2 = self.rbergomi.dW2()

        self.gamma = gamma
        self.Y = rbergomi.Y(self.dW1)

    def run(self, rho0 = -0.5, eta0 = 1.5, maxiter = 10, method = 'L-BFGS-B',
            rho_bnds = (-0.999,0.00), eta_bnds = (0.00,10.00)):
        """
        Method for actually performing the calibration.
        """
        # Begin timer
        t0 = datetime.now()

        self.rho0 = rho0
        self.eta0 = eta0
        self.rho_bnds = rho_bnds
        self.eta_bnds = eta_bnds
        self.maxiter = maxiter
        self.method = method

        # Specify objective function for minimisation
        # Need to control this more: add weighting scheme
        def rmse(x):
            # Assign entries of array x
            rho = x[0]
            eta = x[1]

            # Build appropriate paths for chosen x
            # Assigning instead to self.S etc. slows a small amount
            dZ = self.rbergomi.dB(self.dW1, self.dW2, rho = rho)
            if self.gamma:
                v = self.xi * self.rbergomi.v2(self.Y, eta = eta)
            else:
                v = self.xi * self.rbergomi.V(self.Y, eta = eta)
            S = self.rbergomi.S(v, dZ)

            # Compare implied with actual surface
            # Should change to using rbergomi.IV
            implied = self.rbergomi.surface(S, self.M, self.k)
            rmse = self.rbergomi.rmse(self.actual_surface, implied)
            return rmse

        # Perform calibration
        results =  minimize(rmse, (self.rho0, self.eta0), method = self.method,
                            bounds = (self.rho_bnds, self.eta_bnds),
                            options = {'maxiter': self.maxiter})

        # Assign results to instance
        self.results = results
        self.rho = results.x[0]
        self.eta = results.x[1]
        self.rmse = results.fun

        # Now neatly present outcome
        rmse = 100 * np.round(self.rmse, decimals = 5)
        rho = np.round(self.rho, decimals = 5)
        eta = np.round(self.eta, decimals = 5)
        t1 = datetime.now()
        dt = t1 - t0
        self.time = np.round(dt.seconds + dt.microseconds/1e6, decimals = 3)
        time = self.time
        if results.success == True:
            print('rmse:', rmse)
            print('time:', time)
            print('nit:', results.nit) # Tidy this up
            print('rho:', rho)
            print('eta:', eta)

        else:
            print('Minimum RMSE not found.')

    def paths(self):
        """
        Method for saving v and S paths to the Calibrate instance. Of
        course this uses exact calibration results and same increments.
        """
        # Reconstruct everything with solved parameters
        # This is not very nice: now rbergomi.S is a method and calibrate.S
        # is an array, really we want rbergomi.S to hold the array
        if self.gamma:
            self.v = self.xi * self.rbergomi.v2(self.Y, eta = self.eta)
        else:
            self.v = self.xi * self.rbergomi.V(self.Y, eta = self.eta)

        self.dZ = self.rbergomi.dB(self.dW1, self.dW2, rho = self.rho)
        self.S = self.rbergomi.S(self.v, self.dZ)
        return (self.v, self.S)
