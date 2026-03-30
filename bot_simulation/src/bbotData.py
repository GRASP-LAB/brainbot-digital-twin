import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from itertools import count
from scipy.signal import savgol_filter

import copy
import platform
system_name = platform.node()


rad = lambda x: x*np.pi/180
deg = lambda x: x*180/np.pi
Rot = lambda angle_rad: np.array([[np.cos(angle_rad), -np.sin(angle_rad)],[np.sin(angle_rad), np.cos(angle_rad)]])

cf=0.0006742675003064854 #px to cm
raw_data_path = '../bbot_data/'


#Constant
MAJOR = 81.57/2*cf
MINOR = 44.542/2*cf

rmag = np.sqrt(((1-0.592)*MAJOR)**2+((1-0.390)*MINOR)**2)
Xmax=792*cf
Ymax=718*cf


def dc_fix_angles(df, MOVIE_TIME, delta_t, angle_correction, reverse_direction=0, normalize = False):
    dc = df.copy()
    
    list_i=[]
    
    #fix angles
    if angle_correction == "above180":
        for i in range(len(dc)-1):
            if dc.loc[i, 'Angle']>170:
                if dc.loc[i+1, 'Angle']<10:
                    list_i.append(i+1)
        for i in list_i:
            dc.loc[i:, 'Angle']=dc.loc[i:, 'Angle']+180
    elif angle_correction == "below0":
        for i in range(len(dc)-1):
            if dc.loc[i, 'Angle']<10:
                if dc.loc[i+1, 'Angle']>170:
                    list_i.append(i+1)

        for i in list_i:
            dc.loc[i:, 'Angle']=dc.loc[i:, 'Angle']-180
            
    
    
    dc['Theta'] = (90+dc['Angle']+reverse_direction*180)*np.pi/180
    
    dc = dc[['frame', 'Time', 'X', 'Y','Theta']]
    return dc
    
    
def dc_calculate_vels(dc, delta_t, order='first'):
    #velocities by derivatives
    N = len(dc['X'])
    x = dc['X']
    y = dc['Y']
    theta = dc['Theta']
    
    vx = np.zeros(N)
    vy = np.zeros(N)
    vmag = np.zeros(N)
    w = np.zeros(N)
    
    
    if order=='first':
        
        diff = lambda x, i, dt: (x[i+1]-x[i])/(dt)
        
        for i in range(0, len(dc)-1):
            vx[i] = diff(x, i, delta_t)
            vy[i] = diff(y, i, delta_t)
            w[i] = diff(theta, i, delta_t)
            vmag[i] = np.sqrt(vx[i]**2+vy[i]**2)
        
    elif order=='second':
        vx = np.zeros(N-2)
        vy = np.zeros(N-2)
        vmag = np.zeros(N-2)
        w = np.zeros(N-2)
        diff = lambda x, i, dt: (x[i+1]-x[i-1])/(2*dt)
        
        for i in range(1, len(dc)-1):
            vx[i-1] = diff(x, i, delta_t)
            vy[i-1] = diff(y, i, delta_t)
            w[i-1] = diff(theta, i, delta_t)
            vmag[i-1] = np.sqrt(vx[i-1]**2+vy[i-1])
    
    dc['vx']=vx.copy()
    dc['vy']=vy.copy()
    dc['vmag'] = vmag
    dc['w'] = w.copy()
    
    
    
    return dc
    
def dc_normalize(dc, f1,f2,f3):
    #normalize
   
    
    dc['X'] = dc['X']*f1
    dc['vx'] = dc['vx']*f1
    
    dc['Y'] = dc['Y']*f2
    dc['vy'] = dc['vy']*f2
    
    dc['Theta'] = dc['Theta']*f3
    dc['w'] = dc['w']*f3
    
    return dc

def dc_to_ls(dc):
    dl = dc[['Time', 'X','Y','Theta']]
    dl=dl.rename(columns={'X':'rx', 'Y': 'ry', 'Theta': 'theta'})
    m=len(dc)
    
    dl.loc[:, ['rxf', 'ryf', 'thetaf']] = dl[['rx', 'ry', 'theta']].values
    dl.loc[:,['rxf','ryf', 'thetaf']]=dl.loc[:,['rxf','ryf', 'thetaf']].shift(-1, axis=0).values
    dl = dl.loc[:m-2]
    return dl



def rotation_to_translation(df, dtime,  filtering=True, remove_nans=True):
    if filtering: 
        vpxf = savgol_filter(df.X, 31,5, 1)/dtime
        vpyf = savgol_filter(df.Y, 31,5, 1)/dtime
        wf = savgol_filter(df.Theta, 31,5, 1)/dtime
    else:
        vpxf = df.vx
        vpyf = df.vy
        wf  = df.w
        
    vmagf = np.sqrt(vpxf**2+vpyf**2)
    eta = np.abs((wf*rmag/vmagf))
    if remove_nans:
        etaf = eta[~np.isnan(eta)]
        return etaf
    else:
        return eta


def load():
    original_cwd = os.getcwd()
    rev_direction = np.ones((5,2,9))
    rev_direction[0,0,[0,1,3, 6, 7,8]] = 0
    #(1,0,7) and (1,0,8)???
    rev_direction[1,0,[0,1,2,4,6,]] = 0
    #rev_direction[1,0,[7,8]] = 1

    #(3,0,7) ???
    rev_direction[2,0,[4,7]]=0
    rev_direction[3,0,[0,1,2,4,5,7,8]]=0
    rev_direction[4,0,[1,5,6]]=0


    rev_direction[0,1,[3,5,8]] = 0
    rev_direction[1,1,[0,1,3,5,6,7]] = 0
    rev_direction[2,1,[0,1,4,5,6,7,8]] = 0
    rev_direction[3,1,[0,1,2,3,4,5,6,8]] = 0
    rev_direction[4,1,[0,3,5]] = 0

    os.chdir(raw_data_path)
    cols = ["frame" ,   "Time" ,"Area", "X" ,"Y", "Major", "Minor",  "Angle"] #names for the data
    mps = [20,30,40,50,60,70,80,90,100]
    la = [5,10,15,20,25]
    dfs = [[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]]

    dirr = ['Left', 'Right']

    for i in range(len(la)):
        for j in range(len(mps)):
            filename = '{}°Leg_Left_{}%Pow.txt'.format(la[i], mps[j])
            dfs[i][0].append((pd.read_csv(filename, delimiter='\t', names=cols)))

            filename2 = '{}°Leg_Right_{}%Pow.txt'.format(la[i], mps[j])
            dfs[i][1].append((pd.read_csv(filename2, delimiter='\t', names=cols)))
            #dfL.append(pd.read_csv(filename, delimiter='\t', names=cols))
            #print(cntr, "   ",filename)
            #cntr+=1
    #i - leg angle [0,5)
    #j - left right [0,2)
    #k - [0,9)
    dfs0 = copy.deepcopy(dfs)

    K = len(mps)  # Number of rows 9
    I = len(la)   # Number of columns 5


    movie_ts = [[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]]
    dtime = [[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]]



    for j in range(2):
        for i in range(I):
            for k in range(K):  # Should iterate over columns based on N
                #print(i,j,k)
                data = dfs[i][j][k]
                data['X']*=cf
                data['Y']*=cf
                movie_time = data.Time[len(data.Time)-1]
                delta_time = movie_time/len(data.Time)

                dtime[i][j].append(delta_time)
                movie_ts[i][j].append(movie_time)

                #if i==1 and j==1 and k==K-2:

                if j==0:
                    dfs[i][j][k] = dc_fix_angles(dfs[i][j][k], movie_time, delta_time, angle_correction="below0", reverse_direction=rev_direction[i,j,k])
                elif j==1:
                    dfs[i][j][k] = dc_fix_angles(dfs[i][j][k], movie_time, delta_time, angle_correction="above180", reverse_direction=rev_direction[i,j,k])
                dfs[i][j][k] = dc_calculate_vels(dfs[i][j][k], dtime[i][j][k])

    dfs[1][0][K-2] = dc_fix_angles(dfs0[1][0][K-2], movie_ts[1][0][K-2], dtime[1][0][K-2], angle_correction="above180", reverse_direction=rev_direction[i,j,k])
    dfs[1][0][K-1] = dc_fix_angles(dfs0[1][0][K-1], movie_ts[1][0][K-1], dtime[1][0][K-1], angle_correction="above180", reverse_direction=rev_direction[i,j,k])


    df = lambda i,j,k : dfs[i][j][k]

    dfs[1][0][7].X *= cf
    dfs[1][0][7].Y *= cf


    dfs[1][0][8].X *= cf
    dfs[1][0][8].Y *= cf

    dfs[1][0][7].Theta+=np.pi
    dfs[1][0][8].Theta+=np.pi
    dfs[3][0][7].Theta+=np.pi


    dfs[1][0][8] = dc_calculate_vels(dfs[1][0][8], dtime[1][0][8])   
    dfs[1][0][7] = dc_calculate_vels(dfs[1][0][7], dtime[1][0][7])   


    for j in range(2):
        for i in range(I):
            for k in range(K): 
                dfs[i][j][k]['sigma'] = rotation_to_translation(dfs[i][j][k], dtime[i][j][k], remove_nans=False)
    os.chdir(original_cwd)
    return dfs, dtime, mps, la
    


class Ellipsoid():
    
    id_iter = count()
    def __init__(self, r=np.array([0,0]), theta_rad=rad(0), v=np.array([0,0]), w=0, a=MAJOR, b=MINOR, noise=0, npoints=100, ctend=[1,1]): 
        self.ID = next(Ell.id_iter)
        
        #dimensions
        self.a = a;
        self.b = b;
        self.mass=1

        #position 
        self.r = r # position (x,y)=(r[0], r[1])
              
        #direction\
        t_size = npoints #number of points to plot the ellipsoids
        self.npoints = npoints
        t = np.linspace(0, 2*np.pi, t_size)
        self.theta = theta_rad
        if abs(self.theta)>2*np.pi:
            self.theta=sign(self.theta)*(abs(self.theta)%(2*np.pi))
              
        #ctend limit check
        if ctend[0]<0 or ctend[0]>2 or ctend[1]<0 or ctend[1]>2:
            print("WARNING 1! Physical limits for ctend should be [0,2].")
        if (1-ctend[0])**2+(1-ctend[1])**2>1:
            print("WARNING 2! COM outside Ellipse perimeter.")
           
        #orientation        
        self.body=np.zeros([t_size,2])
        self.body[:,0] = a*np.cos(t)
        self.body[:,1] = b*np.sin(t)
        
        #direction vectors
        self.n1 = np.array([ np.cos(self.theta), np.sin(self.theta)])
        self.n2 = np.array([-np.sin(self.theta), np.cos(self.theta)])
        
        #tendency to center, for the center of mass (com)
        self.ctend = ctend
        
        self.body=self.body@Rot(self.theta).T   #rotation
        self.body+=r                            #translation
        self.c1 = self.r-(1-self.ctend[0])*self.a*self.n1
        self.c2 = self.r-(1-self.ctend[1])*self.b*self.n2
        self.com = self.c1+self.c2-self.r
        
        self.bx = self.body[:,0]
        self.by = self.body[:,1]
        self.hx = self.body[0,0]
        self.hy = self.body[0,1]
        
    
    
if __name__ == "__main__":
    dataLoad()

