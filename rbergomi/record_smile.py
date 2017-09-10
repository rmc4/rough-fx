import os
os.chdir('/Users/ryanmccrickerd/desktop/rbergomi/rbergomi')
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
import xlwings as xw
from rbergomi import rBergomi
from surface import Surface
from routines import *

# % matplotlib inline

n_dict = {'1D':624,
          '1W':104,
          '1M':24,
          '3M':8,
          '6M':4,
          '1Y':2}
# n_dict

tenor = '1Y'
settings = 'both'
n = n_dict[tenor]
T = 2./n
N = 400000
rho = -0.00
alpha = -0.000001

runs = 5

for i in range(runs):

    rbergomi = rBergomi(n = n, N = N, T = T, a = alpha, AS = True)
    surface = Surface(tenor, settings, close = False)

    kwargs = {'xi':0.235**2, 'eta':1.9, 'rho':rho}

    method = base_run
    # method = moment_matching
    # method = conditional_mc
    # method = price_control
    # method = timer_control
    # method = optimal_run

    np.random.seed(0)

    IV = method(rbergomi, surface, **kwargs)

    install_path = '/Users/ryanmccrickerd/desktop'
    file_path = install_path + '/rbergomi/data/market/' + settings + '_' + tenor + '.xlsx'

    # Instantiate xlwings object
    wb = xw.Book(file_path)
    sht = wb.sheets['Sheet1']

    # Paste implied vols
    sht.range('B6').value = 100 * np.flipud(np.transpose(IV))
    sht.range('C6').value = str(i + 1) + ' completed'
    wb.save()

    k = np.array(surface._log_strike_surface())
    MV = np.array(surface.surface)

    for M in range(len(k[:,0])):
        plot, axes = plt.subplots()

        axes.plot(k[M,:], IV[M,:], 'r')

        # 1SD bounds - only correct when not using AV
        # axes.plot(k[M,:], IV[M,:,0], 'k--', linewidth = 0.5) # Lower bound
        # axes.plot(k[M,:], IV[M,:,2], 'k--', linewidth = 0.5) # Upper bound

        axes.plot(k[M,:], MV[M,:], 'go', fillstyle = 'none', ms = 4, mew = 1)

        axes.set_xlabel(r'$k$', fontsize = 14)
        axes.set_ylabel(r'$IV(k,T)$', fontsize = 14)
        axes.set_title(r'$T=$' + surface._tenors[M], fontsize = 14)
        #axes.legend([r'$\mathsf{Standard}$', r'$\mathsf{Timer \ Control}$', r'$\mathsf{S \ Control}$'])
        # plt.xlim([-0.05,0.05])
        # plt.ylim([0.14,0.30])

        plt.grid(True)
