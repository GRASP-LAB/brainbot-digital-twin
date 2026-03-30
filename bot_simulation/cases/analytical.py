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

#constants
A1 = 5.5*0.5/100
A2 = 3.0*0.5/100
rho1 = 0.374
rho2 = 0.661
rmag = np.sqrt((rho1*A1)**2+(rho2*A2)**2)
Xmax = 53.4/100
Ymax = 48.4/100

# Global font settings
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "cm",
    "font.size": 12,
    "font.style": "italic" # Changed from italic to match standard thesis styles
})



#initial position and orientation
r0_init = np.array([Xmax/2,  Ymax/2])
phi_0_init = rad(30)

T = 200
Trot = 2.
dt = 0.01

def run_simulation(fw, fv1, fv2):
    #initial position and orientation
    r0 = r0_init.copy()
    phi_0 = phi_0_init
    b = Ell2D(r = r0, theta_rad = phi_0, vfunc1=fv1, vfunc2=fv2, wfunc=fw)
    df = b.simulate(T,dt, 'leg')
    return df

#Trajectory (a)
fw = lambda t: -1
fv1 = lambda t: 0.02
fv2 = lambda t: 0.005
df0 = run_simulation(fw, fv1, fv2)

#Trajectory (b)
fw  = lambda t: -1.0
fv1 = lambda t:  0.0
fv2 = lambda t:  0.0
df1 = run_simulation(fw, fv1, fv2)


#Trajectory (c)
fw  = lambda t: -1.0
fv1 = lambda t:  0.02
fv2 = lambda t:  0.005
df2 = run_simulation(fw, fv1, fv2)

#Trajectory (d)
fw  = lambda t: -1.0
fv1 = lambda t:  0.001*np.cos(fw(t)*t)
fv2 = lambda t: -0.001*np.sin(fw(t)*t)
df3 = run_simulation(fw, fv1, fv2)

#Trajectory (e)
fw  = lambda t: -1.0
fv1 = lambda t:  0.008+0.001*np.cos(fw(t)*t)
fv2 = lambda t: -0.008-0.001*np.sin(fw(t)*t)
df4 = run_simulation(fw, fv1, fv2)

#Trajectory (f)
fw  = lambda t: -1.0

fv1 = lambda t:  0.008+0.001*np.cos(-1.2*t)
fv2 = lambda t: -0.008-0.001*np.sin(-1.2*t)
df5 = run_simulation(fw, fv1, fv2)

# Global font settings

tr = [df0,df1,df2,df3,df4,df5]
letters = ['(a)', '(b)', '(c)', '(d)', '(e)','(f)', '(g)', '(h)', '(i)', '(j)']
label_fontsize=10
letter_fontsize=10
c=0
size = 8
nh = 3
nw = 6
plt.figure(figsize=(10,5))
lngth = 3500
for j in range(0, nw):  # Three columns
    plt.subplot(nh, nw, j+1)  # Correctly calculate the index
    plt.plot(tr[c].X[:lngth]*100, tr[c].Y[:lngth]*100, 'C3', zorder=100)
    plt.plot(tr[c].Gicrx[:lngth]*100, tr[c].Gicry[:lngth]*100, 'green', zorder=100)
    plotBot(tr[c].X[0], tr[c].Y[0], tr[c].Theta[0], scale='cm')
    plt.axis('equal')
    plt.xlabel('x (cm)')
    plt.text(0.05, 0.95, letters[c], transform=plt.gca().transAxes, fontsize=letter_fontsize, verticalalignment='top')
    if j==0:
        plt.ylabel('y (cm)',)
    
    plt.subplot(3, 6, j+nw+1) 
    plt.plot(tr[c].time[:lngth], tr[c].sigma[:lngth])
    plt.ylim([-0.2,2.2])
    if j==0:
        plt.ylabel(r'$\eta$',)
    plt.xlabel('t (s)')

    
    plt.subplot(3, 6, j+2*nw+1)

    data=tr[c].sigma

    if c>2:
        plt.hist(data, bins=100, color='gray', density=True, stacked=True)
        plt.xlabel(r"$\eta$")
    else:
        plt.hist(df1.sigma*np.average(data), bins=1, color='gray', edgecolor='k',  density=True, stacked=True)
        plt.xlabel(r"$\eta$")
        plt.yticks([])
    if j==0:
        plt.ylabel(r"$P(\eta)$")
    c+=1

plt.tight_layout()
plt.savefig('../code_outputs/figures/deterministic_traj2.pdf', dpi=300)