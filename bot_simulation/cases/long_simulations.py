
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



def remove_outliers(data, method="iqr", k=1.5):
    if method == "std":
        mu, std = np.mean(data), np.std(data)
        return data[(data > mu - k*std) & (data < mu + k*std)]
    elif method == "iqr":
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - k*iqr, q3 + k*iqr
        return data[(data >= lower) & (data <= upper)]
    elif method == "percentile":
        lower, upper = np.percentile(data, [k, 100-k])
        return data[(data >= lower) & (data <= upper)]
    else:
        return data  # no filtering

plt.figure(figsize=(12,4))
normal_dists = []
skewed_dists = []

for i in range(4):
    Xf, Yf, Thetaf, vpxf, vpyf, wf = filter_XY(trajectories[i], dts[i], True, 7, 2)
    w0 = wf#trajectories[i].w # raw data
    # remove outliers (choose method: "iqr", "std", or "percentile")
    w0_clean = remove_outliers(w0, method="none", k=1.5)
    # fit distributions
    a, loc, scale = skewnorm.fit(w0_clean)
    mu, std = norm.fit(w0_clean)
    # build frozen distributions
    normal_dists.append(norm(mu, std))
    skewed_dists.append(skewnorm(a, loc, scale))
    # x-range for smooth pdf
    wrange = np.linspace(np.min(w0_clean), np.max(w0_clean), 200)

    # plot histogram + PDFs
    # plt.subplot(1,4,i+1)
    # plt.hist(w0, bins=60, density=True, alpha=0.3, color="gray", label="Original")
    # plt.hist(w0_clean, bins=60, density=True, alpha=0.6, color="blue", label="Filtered")
    # plt.plot(wrange, normal_dists[i].pdf(wrange), "k-", lw=2, label="Normal fit")
    # plt.plot(wrange, skewed_dists[i].pdf(wrange), "r--", lw=2, label="Skew-normal fit")


from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec

# GridSpec updated to 6 rows
fig = plt.figure(figsize=(10, 14))
gs = gridspec.GridSpec(6, 4, hspace=0.45, wspace=0.3)
plot_type = 0

for tr_id in range(4):
    df = trajectories[tr_id]
    edt = dts[tr_id]
    vo_1, vo_2, vh_1, vh_2, vh_3, vh_4 = params[tr_id]
    Xf, Yf, Thetaf, vpxf, vpyf, wf = filter_XY(df, edt, True, 7, 2)
    vpmag = np.sqrt(vpxf**2+vpyf**2)
    tt = df.Time
    dt = 0.03
    T = np.max(df.Time)*10
    
    accumulated_w = []
    accumulated_vpmag = [] # New accumulator for Row 4
    accumulated_sigma = []

    for cnt in range(10):
        np.random.seed(cnt+100)
        N = int(T/dt) + 1
        random_w = skewed_dists[tr_id].rvs(size=N)
        fw = lambda t: random_w[int(t/dt)] if t < T else 0
        
        fv1 = lambda t: vo_1 + vh_1 * np.cos(t) - vh_2 * np.sin(t)
        fv2 = lambda t: vo_2 - vh_3 * np.cos(t) - vh_4 * np.sin(t)
        b0 = BBot(r=np.array([Xf[0], Yf[0]]), theta_rad=Thetaf[0],
                    vfunc1=fv1, vfunc2=fv2, wfunc=fw)

        df3 = simulate_bbot(b0, T, dt, 'CN', dependency='angle')
        b0.df = df3
        b0.postprocess()

        accumulated_w.extend(random_w[:len(df3)])
        accumulated_vpmag.extend(df3.vpmag) # Accumulating for Row 4
        accumulated_sigma.extend(df3.sigma)

        if cnt == 0:
            ax0     = fig.add_subplot(gs[0, tr_id])  # Row 0: copy of traj
            ax1     = fig.add_subplot(gs[1, tr_id])  # Row 1: traj
            ax3     = fig.add_subplot(gs[2, tr_id])  # Row 2: angular velocity time series
            ax2     = fig.add_subplot(gs[3, tr_id])  # Row 3: P(ω)
            ax5     = fig.add_subplot(gs[4, tr_id])  # Row 4: new placeholder P(ω)
            ax4     = fig.add_subplot(gs[5, tr_id])  # Row 5: P(η)
            ax_inset = inset_axes(ax3, width="45%", height="40%", loc=3, borderpad=-1.4)

        ax0.plot(df3.X[:int(23/dt)]*100, df3.Y[:int(23/dt)]*100, lw=1)
        ax1.plot(df3.X*100, df3.Y*100, lw=1)

    ax0.axis('equal')
    ax1.axis('equal')

    # Row 2: angular velocity time series
    ax3.plot(tt, wf, label='Original ω', c='black')
    ax3.plot(df3.time, random_w[:len(df3)], label='Filtered ω', c='C3', alpha=0.6)

    # Inset on ax3
    ax_inset.plot(tt, wf, c='black', lw=0.5)
    ax_inset.plot(df3.time[:int(len(df))], random_w[:int(len(df))], c='C3', alpha=0.5, lw=0.5)
    ax_inset.tick_params(axis='both', which='both', labelsize=6)

    if plot_type == 0:
        wrange = np.linspace(np.min(wf), np.max(wf), 200)

        # Row 3: P(ω) - Now Ensemble Averaged
        ax2.hist(wf, bins=60, density=True, color="black")
        ax2.hist(accumulated_w, bins=60, density=True, color="C3", alpha=0.) # Added ensemble
        ax2.plot(wrange, skewed_dists[tr_id].pdf(wrange), "r--", lw=2)

        # Row 4: P(v) - Now Ensemble Averaged
        ax5.hist(vpmag, bins=60, density=True, color="black")
        ax5.hist(accumulated_vpmag, density=True, bins=60, color='C3', alpha=0.6) # Changed from df3.vpmag

        # Row 5: P(η) - Ensemble Averaged
        ax4.hist(np.abs(wf/vpmag*1.45/100), density=True, bins=60, color='black')
        ax4.hist(accumulated_sigma, density=True, bins=60, color='C3', alpha=0.6)
    else:
        ax2.plot(tt, wf, c='black')
        ax5.plot(tt, wf, c='black')  # placeholder — replace later
        ax4.plot(df.Time, np.abs(wf/vpmag*rmag), 'black')

    # x-labels
    ax1.set_xlabel("x (cm)")
    ax3.set_xlabel(r't (s)')
    ax2.set_xlabel(r"$\omega$ (rad/s)")
    ax5.set_xlabel(r"$\omega$ (rad/s)")
    ax4.set_xlabel(r"$\eta$")

    # y-labels (leftmost column only)
    if tr_id == 0:
        ax0.set_ylabel("y (cm)")
        ax1.set_ylabel("y (cm)")
        ax3.set_ylabel(r"$\omega$ (rad/s)")
        ax2.set_ylabel(r"$P(\omega)$")
        ax5.set_ylabel(r"$P(v)$")
        ax4.set_ylabel(r"$P(\eta)$")

plt.savefig(f'../code_outputs/figures/multiple_noise_inset_{tr_id}.pdf', dpi=300)