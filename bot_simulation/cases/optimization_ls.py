
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
from scipy.optimize import least_squares, minimize

DPI = 300

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "cm",
    "font.size": 12,
    "font.style": "italic" # Changed from italic to match standard thesis styles
})



def error_function_general(sim_params, id, erf_id,
                           data, dt, theta_interpolator, w_interpolator, 
                           direction, optype='residual', sim_order=1, fw_type='interpolate'):
    time, X,Y,T, wf,Rx,Ry, etaf = data
    x0, y0, phi_0 = X[id[0]], Y[id[0]], T[id[0]]
    t_0 = time[id[0]]
    t_f = time[id[1]]
    sim_time = t_f-t_0
    fv1, fv2 = erfs[erf_id](sim_params, theta_interpolator,w_interpolator, direction)
    if sim_order==1:
        b0 = Ell2D(r = np.array([x0,y0]), theta_rad = phi_0, t0=t_0, vfunc1=fv1, vfunc2=fv2, wfunc=w_interpolator)
        df_sim = b0.simulate(sim_time, dt, 'leg')
    elif sim_order==2:
        if fw_type == 'interpolate':
            b0 = BBot(r = np.array([x0,y0]), theta_rad = phi_0, t0=t_0, vfunc1=fv1, vfunc2=fv2, wfunc=w_interpolator)
            df_sim = simulate_bbot(b0, sim_time, dt, 'CN', 'time', False)
        elif fw_type == 'fft':
            b0 = BBot(r = np.array([x0,y0]), theta_rad = phi_0, t0=t_0, vfunc1=fv1, vfunc2=fv2, wfunc=w_interpolator)
            df_sim = simulate_bbot(b0, sim_time, dt, 'CN', 'angle', False)



    residual_x=(X[id[0]:id[1]] - df_sim.X)
    residual_y=(Y[id[0]:id[1]] - df_sim.Y)
    
    if optype == 'residual':
        return np.concatenate([residual_x, residual_y,])  # 
    elif optype == 'euclidian':
        return np.sum(np.sqrt(residual_x**2+residual_y**2))



def optimize_ls(i, j, k, erf_id, normalize=False, optype='residual', fw_type = 'interpolate', sim_order=1):
    edt = dtime[i][j][k]
    time = dfs[i][j][k].Time
    df = dfs[i][j][k]
    dirr = np.sign(np.average((df.w)))
    Xf, Yf, Thetaf, vpxf, vpyf, wf = filter_XY(df, edt, True, 21, 3)

    # Compute trajectory components
    v1f = vpxf * np.cos(Thetaf) + vpyf * np.sin(Thetaf)
    v2f = -vpxf * np.sin(Thetaf) + vpyf * np.cos(Thetaf)
    rho1 = 0.374
    rho2 = dirr * 0.661
    vc1 = v1f - wf * rho2 * A2
    vc2 = v2f - wf * rho1 * A1
    Rx = -vpyf / wf
    Ry = vpxf / wf
    rx = Xf - vpyf / wf
    ry = Yf + vpxf / wf
    traj_radius = np.sqrt(vpxf**2 + vpyf**2) / np.abs(wf)
    etaf = savgol_filter(df.w/df.vmag, 21,3)*1.45/100#wf*1.45/100/np.sqrt(vpxf**2+vpyf**2)#savgol_filter(df.w/df.vmag, 21,3)*1.45/100#
    data = [time, Xf, Yf, Thetaf, wf, rx, ry, etaf]
    df_opt = []

    vs = []
    l0 = 0
    l1 = len(df) - 1
    sim_time = time[l1] - time[l0]

    # Set up interpolators for angular variables
    if fw_type=='interpolate':
        fw_interp = interp1d(time, wf, kind='cubic', fill_value="extrapolate")
        theta_interp = interp1d(time, Thetaf, kind='cubic', fill_value="extrapolate")
        fw = lambda t: fw_interp(t).item()
        theta_func = lambda t: theta_interp(t).item()
    elif fw_type=='fft':
        print('none')
        

    # Error function closure
    error_compute = lambda params: error_function_general(
        params, [l0, l1], erf_id, data, edt, theta_interp, fw, dirr, optype, sim_order, fw_type
    )

    # Choose optimizer based on optype
    if optype == 'residual':
        result = least_squares(error_compute, np.random.rand(erf_param_len[erf_id]))
    elif optype == 'euclidian':
        result = minimize(error_compute, np.random.rand(erf_param_len[erf_id]), method='L-BFGS-B')

    optimized_params = result.x
    fv1, fv2 = erfs[erf_id](optimized_params, theta_interp, fw_interp, dirr)
    print(result.x, np.round(result.fun if optype == 'euclidian' else result.cost, 5))

    vs.append(result.x)


    if sim_order==1:
        b0 = Ell2D(r=[Xf[l0], Yf[l0]], theta_rad=Thetaf[l0], t0=time[l0],
                     vfunc1=fv1, vfunc2=fv2, wfunc=fw)
        df_sim = b0.simulate(sim_time, edt, 'leg')
    elif sim_order==2: 
        if fw_type=='interpolate':
            b0 = BBot(r=[Xf[l0], Yf[l0]], theta_rad=Thetaf[l0], t0=time[l0],
                        vfunc1=fv1, vfunc2=fv2, wfunc=fw)
            
            df_sim  = simulate_bbot(b0, sim_time, edt, 'CN', 'time', False)
        if fw_type=='fft':
            b0 = BBot(r=[Xf[l0], Yf[l0]], theta_rad=Thetaf[l0], t0=time[l0],
                        vfunc1=fv1, vfunc2=fv2, wfunc=fw)
            
            df_sim  = simulate_bbot(b0, sim_time, edt, 'CN', 'angle', False)
            b0.df=df_sim
            b0.postprocess()
    df_opt.append(df_sim)

    return data, df_sim, result, vs


def compute_least_squares_cost(X, Y, Xfit, Yfit):
    residual_x = X - Xfit
    residual_y = Y - Yfit
    residuals = np.concatenate([residual_x, residual_y])
    cost_i = np.sqrt(residual_x ** 2+residual_y**2)
    cost = 0.5 * np.sum(residuals ** 2)
    return cost_i, cost

edatas = []
opts   = []
ress   = []
errors = []
vparams = []
rows = 0
ijk = [ [2,0,8],[4,1,5], [0,0,5],[2,1,7]]
models = [11,11,11,11]
for c in range(4):    
    rows+=1
    i,j,k = ijk[c]
    df = dfs[i][j][k]
    edata,  opt8, res,vs = optimize_ls(i,j,k, 11, 'residual', fw_type='interpolate', sim_order=2)
    edatas.append(edata)
    opts.append(opt8)
    ress.append(res)
    vparams.append(vs)
    l1  = len(edata[1])-1


errors = np.array(errors)

fig, axs = plt.subplots(2, 4, figsize=(10, 5))  # 2 rows, 5 columns
fig.subplots_adjust(hspace=0.4, wspace=0.3)  # vertical spacing

cols = 4
errors = []
params = []
letters = 'abcdefghijkl'
shift=0
shift_delta = lambda x: 1.01*(np.max(x)-np.min(x))
cntr=0

for c in range(len(ijk)):
    i,j,k = ijk[c]
    df = dfs[i][j][k]
    edata = edatas[c]
    opt8 = opts[c]
    res = ress[c]
    vs = vparams[c]

    xrange = [df.X.min() * 100, df.X.max() * 100]
    yrange = [df.Y.min() * 100, df.Y.max() * 100]
    delta = max(xrange[1] - xrange[0], yrange[1] - yrange[0])
    xrange[1] = xrange[0] + delta
    yrange[1] = yrange[0] + delta

    ax_top = axs[0, c]
    ax_bottom = axs[1, c]

    ax_top.plot(edata[1]*100, edata[2]*100, c='black', lw=1)
    ax_top.plot((opt8.X + shift_delta(opt8.X) * shift)*100, opt8.Y*100, c='C3')
    ax_top.set_xlabel('x (cm)')
    if c == 0:
        ax_top.set_ylabel('y (cm)', )
    ax_top.axis('equal')
    ax_top.text(0.85, 0.9, f'({letters[c]})', transform=ax_top.transAxes)

    ax_bottom.plot(edata[0], np.abs(edata[7]), c='black', lw=1)
    ax_bottom.plot(opt8.time, opt8.sigma, c='C3')
    ax_bottom.set_xlabel('t (s)')
    if c == 0:
        ax_bottom.set_ylabel(r'$\eta$')
    ax_bottom.text(0.866, 0.9, f'({letters[c+4]})', transform=ax_bottom.transAxes)
    ax_bottom.set_ylim([-0.1, 2])


fig.savefig(f'../code_outputs/figures/constopt_4series2_shifted{shift}_.pdf', dpi=600)
