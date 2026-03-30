
import sys
import os
sys.path.append(os.path.abspath('..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from itertools import count
import random

print(os.getcwd())
#user defined
from src.Ellipse2dObj import *
from src.animate import *
from src.trajectory import *
from src.simopt import *
from src.optimizer_utils import *

#load data
from src import bbotData
#from bbotData import Ellipsoid
dfs, dtime, mps, la = bbotData.load()
sys.path.append(os.path.abspath('..'))
from scipy.interpolate import interp1d
from scipy.stats import norm, skewnorm

#constants
A1 = 5.5*0.5/100
A2 = 3.0*0.5/100
rho1 = 0.374
rho2 = 0.661
rmag = np.sqrt((rho1*A1)**2+(rho2*A2)**2)
Xmax = 53.4/100
Ymax = 48.4/100

DPI = 300

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "cm",
    "font.size": 12,
    "font.style": "italic" # Changed from italic to match standard thesis styles
})



trajectories = [dfs[2][0][8], dfs[4][1][5], dfs[0][0][5], dfs[2][1][7]]
dts = [dtime[2][0][8], dtime[4][1][5], dtime[0][0][5], dtime[2][1][7]]


params0 = np.array([-1.09955146e-05,  2.46417882e-03, 1.18788379e-05, 
                    -2.00987818e-03, -3.00239446e-05, 1.82371288e-04]) #0.00027
params1 = np.array([ 0.01751109,  0.01644555, -0.00014531, 
                    -0.00248091, -0.00136197,  0.00176313]) #0.00402

params2 = np.array([ 0.02325366, -0.01146961,  0.00374689,
                     0.00275142, -0.00171661,  0.00262916]) #0.00953

params3 = np.array([ 0.02615222, -0.02379747,  0.00346216,
                     0.00253308,  0.02139487,  0.02729881]) #0.00034

params = [params0, params1, params2, params3]
thresholds = [[0.1, 0.0385, 0.035, 0.0],
              [0.1, 0.05, 0.01, 0.001],
              [0.2, 0.15, 0.05,  0.01],
              [0.1, 0.02, 0.01, 0.001 ]]

thresholds_f = [[2.5, 2.5, 2.5, 2.5],
               [2.5, 2.5, 2.5, 2.5],
               [2.5, 2.5, 2.5, 2.5],
               [2.5, 2.5, 2.5, 2.5]]




import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

n_modes = [1, 3, 11, 69]
# Assume df, edt, dff, filter_XY, BBot, simulate_bbot are already defined
for tr_id in range(2,3):
    df = trajectories[tr_id]
    edt = dts[tr_id]

    vo_1, vo_2, vh_1, vh_2, vh_3, vh_4 = params[tr_id]
    Xf, Yf, Thetaf, vpxf, vpyf, wf = filter_XY(df, edt, True, 7, 2)
    vpmag = np.sqrt(vpxf**2+vpyf**2)
    tt = df.Time

    dt = 0.03
    T = np.max(df.Time)
    n = len(thresholds)
    # Create a 2-row, n-column figure (equal column width)
    #fig = plt.figure(figsize=(4 * n, 12))  # width scales with n
    fig = plt.figure(figsize=(10, 10))  # width scales with n

    gs = gridspec.GridSpec(4, n, hspace=0.4, wspace=0.3)  # horizontal layout

    for i, threshold in enumerate(thresholds[tr_id]):
        # Frequency-domain angular velocity function
        fw = lambda_fft(dts[tr_id], df.Time,  wf, threshold, fmin = 0, fmax=thresholds_f[tr_id][i], plotting=False)
        print(thresholds_f[tr_id][i])
        fv1 = lambda t:  vo_1 + vh_1 * np.cos(t) - vh_2 * np.sin(t)
        fv2 = lambda t:  vo_2 - vh_3 * np.cos(t) - vh_4 * np.sin(t)

        # Simulate bbot
        b0 = BBot(r=np.array([Xf[0], Yf[0]]), theta_rad=Thetaf[0],
                vfunc1=fv1, vfunc2=fv2, wfunc=fw)

        df3 = simulate_bbot(b0, T, dt, 'CN', dependency='angle')
        b0.df = df3
        b0.postprocess()

        # Top row: trajectory
        ax1 = fig.add_subplot(gs[0, i])
        ax2 = fig.add_subplot(gs[1, i])

        #shift X and Y for better scales
        exp_X = (df.X-0.2)*100
        exp_Y = (df.Y-0.1)*100
        sim_X = (df3.X-0.2)*100
        sim_Y = (df3.Y-0.1)*100

        ax1.plot(exp_X, exp_Y, c='k', lw=1)
        ax1.plot(sim_X, sim_Y, c='C3', lw=1)
        #ax1.set_title(f"Threshold = {threshold}")
        ax1.set_ylim([0,30])
        ax1.set_xlim([0,30])
        #ax1.axis('equal')

        # Bottom row: angular velocity
        ax2.plot(tt, wf, label='Original ω', c='black')
        ax2.plot(tt, fw(tt), label='Filtered ω', c='C3')
        ax2.text(0.05, 0.1, r"$N_{modes}$"+f"$={n_modes[i]}$",  transform=ax2.transAxes)

        #3rd row
        ax3 = fig.add_subplot(gs[2, i])
        ax3.plot(df.Time, np.abs(wf/vpmag*1.45/100), 'black')
        ax3.plot(df3.time, df3.sigma, 'C3')

        ax4 = fig.add_subplot(gs[3, i])
        ax4.hist(np.abs(wf/vpmag*rmag), density=True, bins=60, color='black', stacked=False)
        ax4.hist(df3.sigma, density=True, bins=60, color='C3', alpha=0.6)

        #ax2.legend(fontsize=8)
        ax3.set_ylim([-0.1,1.5])
        #labels
        ax1.set_xlabel("x (cm)")
        ax2.set_xlabel("t (s)")
        ax3.set_xlabel("t (s)")
        ax4.set_xlabel(r"$\eta$")

        if i==0:
            ax1.set_ylabel("y (cm)")
            ax2.set_ylabel(r"$\omega$ (rad/s)")
            ax3.set_ylabel(r"$\eta$")
            ax4.set_ylabel(r"$P(\eta)$")
        #write out the data
        tr_names = ['fft(a).txt','fft(b).txt','fft(c).txt','fft(dd).txt','fft(ee).txt','fft(f).txt',]
        save_path = '../code_outputs/data/'

        dd = df3.rename({'time': 'time(s)', 'X': 'rx(cm)', 'Y': 'ry(cm)', 'Theta':'phi(rad)', 'Gicrx': 'rcx(cm)', 'Gicry': 'rcy(cm)', 'sigma': 'eta'}, axis=1)
        dd['rx(cm)'] = dd['rx(cm)'] *100
        dd['ry(cm)'] = dd['ry(cm)'] *100
        dd['rcx(cm)'] = dd['rcy(cm)'] *100
        dd['rcx(cm)'] = dd['rcy(cm)'] *100

        dd = dd.iloc[:, [0,1,2,3, 20, 21,19,11]]

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(save_path + tr_names[i], 'w') as f:
            print(save_path)
            f.write(dd.to_string(index=False, float_format='{:>10.5f}'.format))
            

        
    #plt.tight_layout()
    plt.savefig(f'../code_outputs/figures/fourier_filtered_{tr_id}_modes.pdf', dpi=300)
