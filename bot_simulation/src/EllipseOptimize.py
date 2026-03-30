import numpy as np
import pandas as pd
import scipy.optimize



def extend_array(array, m):
    new_array = np.zeros(len(array)*m)
    for i in range(len(array)):
        for j in range(m):
            new_array[i*m+j] = array[i]
            
    return new_array
    
    
    
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



#3 dof
def Residual(v,w,gamma,major,dt, DF, idx):
    gA = (1-gamma)*major
    
    
    if idx=='all':
        px_k = DF['rx']
        px_k2 = DF['rxf']
        py_k =  DF['ry']
        py_k2 = DF['ryf']
        phi_k = DF['theta']
        phi_k2 = DF['thetaf']
    else:
        px_k = DF.loc[idx,'rx']
        px_k2 = DF.loc[idx,'rxf']
        py_k =  DF.loc[idx,'ry']
        py_k2 = DF.loc[idx,'ryf']
        phi_k = DF.loc[idx,'theta']
        phi_k2 = DF.loc[idx,'thetaf']
    
    
    R1 = px_k-px_k2+gA*(np.cos(phi_k+w*dt)-np.cos(phi_k))+v*dt*np.cos(phi_k)
    R2 = py_k-py_k2+gA*(np.sin(phi_k+w*dt)-np.sin(phi_k))+v*dt*np.sin(phi_k)
    R3 = phi_k-phi_k2+w*dt
    
    res = np.array([R1,R2, R3]).T
    return res


    

#3 dof
def Jacobian(v,w,gamma, major, dt, DF, idx):
    gA = (1-gamma)*major
    
    if idx=='all':
        m = len(DF['theta'])
        J = np.zeros([m*3,3])
        
        for i in range(m):
            #phi = DF.loc[i,'theta']
            phi = DF.loc[i,'theta']
            J[i*3,0] = dt*np.cos(phi)
            J[i*3,1] = -dt*gA*np.sin(phi+w*dt)
            J[i*3,2] = -major*(np.cos(phi+w*dt)-np.cos(phi))

            J[i*3+1,0] = dt*np.sin(phi)
            J[i*3+1,1] = dt*gA*np.cos(phi+w*dt)
            J[i*3+1,2] = -major*(np.sin(phi+w*dt)-np.sin(phi))
    
            J[i*3+2,1] = dt
        
    else:
        J = np.zeros([3,3])
        phi = DF.loc[idx,'theta']
        J[0,0] = dt*np.cos(phi)
        J[0,1] = -dt*gA*np.sin(phi+w*dt)
        J[0,2] = -major*(np.cos(phi+w*dt)-np.cos(phi))
    
        J[1,0] = dt*np.sin(phi)
        J[1,1] = dt*gA*np.cos(phi+w*dt)
        J[1,2] = -major*(np.sin(phi+w*dt)-np.sin(phi))
    
        J[2,1] = dt

    return J
    


    
    
    
#example Objective function and Jacobian for optimization function
#F_obj = lambda u: Residual(u[0], u[1], u[2], MAJOR, dtime, DATA, idx='all').flatten()
#Jac = lambda u: Jacobian(u[0], u[1], u[2], MAJOR, dtime, DATA, idx='all')
#LS = scipy.optimize.least_squares(F_obj, x0=u0, bounds=bounds, jac=Jac)
#bounds_v = (-100,100)
#bounds_w = (-30,30)
#bounds_g = (0,1)
#bounds = [(bounds_v[0],bounds_w[0],bounds_g[0]),(bounds_v[1],bounds_w[1],bounds_g[1])]

#u0 = (0,1,1)


#LS = scipy.optimize.least_squares(F_obj, x0=u0, bounds=bounds, jac=Jac)
#LS.x


def LS_chunks(kpoints, data_ls):
    vs, ws, gs = [],[],[]
    dtime = data_ls.loc[1,'Time']-data_ls.loc[0, 'Time']
    
    bounds_v = (-1000,1000)
    bounds_w = (-30,30)
    bounds_g = (0,1)
    bounds = [(bounds_v[0],bounds_w[0],bounds_g[0]),(bounds_v[1],bounds_w[1],bounds_g[1])]
    
    MAJOR = 81.57/2
    MINOR = 44.542/2
    
    for i in range(int(len(data_ls)/kpoints)):
        DATA = data_ls[i*kpoints : i*kpoints+kpoints]
        DATA = DATA.reset_index()
        #Objective function
        F_obj = lambda u: Residual(u[0], u[1], u[2], MAJOR, dtime, DATA, idx='all').flatten()
        Jac = lambda u: Jacobian(u[0], u[1], u[2], MAJOR, dtime, DATA, idx='all')

        u0 = (0,1,1)

        LS = scipy.optimize.least_squares(F_obj, x0=u0, bounds=bounds, jac=Jac)
        vs.append(LS.x[0])
        ws.append(LS.x[1])
        gs.append(LS.x[2])


    vs = np.array(vs)
    ws = np.array(ws)
    gs = np.array(gs)
    ks = np.arange(len(vs))
    
    return ks,vs,ws,gs
    
    
    

    
    
#----additions November 4, 2023

#----------5DOF
def Res5(vx, vy, w, gamma1, gamma2, major, minor, dt, DF, idx):  
    if idx=='all':
        px_k = DF['rx']
        px_k2 = DF['rxf']
        py_k =  DF['ry']
        py_k2 = DF['ryf']
        phi_k = DF['theta']
        phi_k2 = DF['thetaf']
        
    else:
        px_k = DF.loc[idx,'rx']
        px_k2 = DF.loc[idx,'rxf']
        py_k =  DF.loc[idx,'ry']
        py_k2 = DF.loc[idx,'ryf']
        phi_k = DF.loc[idx,'theta']
        phi_k2 = DF.loc[idx,'thetaf']
        
    
    a = (1-gamma1)*major
    b = (1-gamma2)*minor
    
    R1 = -px_k2 +px_k +a*(np.cos(phi_k+w*dt)-np.cos(phi_k))-b*(np.sin(phi_k+w*dt)-np.sin(phi_k))+vx*dt
    R2 = -py_k2 +py_k +a*(np.sin(phi_k+w*dt)-np.sin(phi_k))+b*(np.cos(phi_k+w*dt)-np.cos(phi_k))+vy*dt
    R3 = -phi_k2+phi_k+w*dt
    
    
    # old approach 
    #cx = px_k-a*np.cos(phi_k)+b*np.sin(phi_k) #a*np.cos(phi_k)-b*np.sin(phi_k)
    #cy = py_k-a*np.sin(phi_k)-b*np.cos(phi_k) #a*np.sin(phi_k)+b*np.cos(phi_k)
    #R1 = -px_k2+(px_k-cx)*np.cos(w*dt)-(py_k-cy)*np.sin(w*dt)+cx+vx*dt
    #R2 = -py_k2+(px_k-cx)*np.sin(w*dt)+(py_k-cy)*np.cos(w*dt)+cy+vy*dt
    #R3 = -phi_k2+phi_k+w*dt
    
    res = np.array([R1,R2,R3]).T
    
    return res

def Jac5(vx, vy, w, gamma1, gamma2, major, minor, dt, DF, idx):

    a = (1-gamma1)*major
    b = (1-gamma2)*minor
    
    if idx=='all':
        m = len(DF['theta'])
        J = np.zeros([m*3,5])
        
        for i in range(m):
            #phi = DF.loc[i,'theta']
            phi_k = DF.loc[i,'theta']
            px_k = DF.loc[i,'rx']
            py_k =  DF.loc[i,'ry']
            
            
            J[i*3,0] = dt
            #J[i*3,1] = 0
            J[i*3,2] = -dt*(a*np.sin(phi_k+w*dt)+b*np.cos(phi_k+w*dt))
            J[i*3,3] = -major*(np.cos(phi_k+w*dt)-np.cos(phi_k))
            J[i*3,4] =  minor*(np.sin(phi_k+w*dt)-np.sin(phi_k))
            
            #J[i*3+1,0] = 0
            J[i*3+1,1] = dt
            J[i*3+1,2] = dt*(a*np.cos(phi_k+w*dt)-b*np.sin(phi_k+w*dt))
            J[i*3+1,3] = -major*(np.sin(phi_k+w*dt)-np.sin(phi_k))
            J[i*3+1,4] = -minor*(np.cos(phi_k+w*dt)-np.cos(phi_k))
    
            J[i*3+2,2] = dt
        
    else:
        J = np.zeros([3,5])
        phi_k = DF.loc[idx,'theta']
        px_k = DF.loc[idx,'rx']
        py_k =  DF.loc[idx,'ry']
        
        
        J[0,0] = dt
        #J[0,1] = 0
        J[0,2] = -dt*(a*np.sin(phi_k+w*dt)+b*np.cos(phi_k+w*dt))
        J[0,3] = -major*(np.cos(phi_k+w*dt)-np.cos(phi_k))
        J[0,4] =  minor*(np.sin(phi_k+w*dt)-np.sin(phi_k))
    
        #J[1,0] = 0
        J[1,1] = dt
        J[1,2] = dt*(a*np.cos(phi_k+w*dt)-b*np.sin(phi_k+w*dt))
        J[1,3] = -major*(np.sin(phi_k+w*dt)-np.sin(phi_k))
        J[1,4] = -minor*(np.cos(phi_k+w*dt)-np.cos(phi_k))
    
        J[2,2] = dt
        
    

    return J
    
#------------3DOF
    
def Res3(vx, vy, w, gamma1, gamma2, major, minor, dt, DF, idx):  
    if idx=='all':
        px_k = DF['rx']
        px_k2 = DF['rxf']
        py_k =  DF['ry']
        py_k2 = DF['ryf']
        phi_k = DF['theta']
        phi_k2 = DF['thetaf']
        
    else:
        px_k = DF.loc[idx,'rx']
        px_k2 = DF.loc[idx,'rxf']
        py_k =  DF.loc[idx,'ry']
        py_k2 = DF.loc[idx,'ryf']
        phi_k = DF.loc[idx,'theta']
        phi_k2 = DF.loc[idx,'thetaf']
        
    
    
    a = (1-gamma1)*major
    b = (1-gamma2)*minor
    
    px_calc = px_k +a*(np.cos(phi_k+w*dt)-np.cos(phi_k))-b*(np.sin(phi_k+w*dt)-np.sin(phi_k))+vx*dt
    py_calc = py_k +a*(np.sin(phi_k+w*dt)-np.sin(phi_k))+b*(np.cos(phi_k+w*dt)-np.cos(phi_k))+vy*dt
    phi_calc = phi_k+w*dt
    
    R1 = -px_k2 + px_calc
    R2 = -py_k2 + py_calc
    R3 = -phi_k2+ phi_calc
    
    
    #old approach 
    #cx = px_k-a*np.cos(phi_k)+b*np.sin(phi_k) #a*np.cos(phi_k)-b*np.sin(phi_k)
    #cy = py_k-a*np.sin(phi_k)-b*np.cos(phi_k) #a*np.sin(phi_k)+b*np.cos(phi_k)
    #R1 = -px_k2+(px_k-cx)*np.cos(w*dt)-(py_k-cy)*np.sin(w*dt)+cx+vx*dt
    #R2 = -py_k2+(px_k-cx)*np.sin(w*dt)+(py_k-cy)*np.cos(w*dt)+cy+vy*dt
    #R3 = -phi_k2+phi_k+w*dt
    
    res = np.array([R1,R2,R3]).T
    return res

def Jac3(vx, vy, w, gamma1, gamma2, major, minor, dt, DF, idx):
    a = (1-gamma1)*major
    b = (1-gamma2)*minor
    if idx=='all':
        m = len(DF['theta'])
        J = np.zeros([m*3,3])
        
        for i in range(m):
            #phi = DF.loc[i,'theta']
            phi_k = DF.loc[i,'theta']
            px_k = DF.loc[i,'rx']
            py_k =  DF.loc[i,'ry']
            cx = px_k+(1-gamma1)*major*np.cos(phi_k)-(1-gamma2)*minor*np.sin(phi_k)
            cy = py_k+(1-gamma1)*major*np.sin(phi_k)+(1-gamma2)*minor*np.cos(phi_k)
            
            J[i*3,0] = dt
            #J[i*3,1] = 0
            #J[i*3,2] = -dt*(a*np.sin(phi_k+w*dt)+b*np.cos(phi_k+w*dt))
            #J[i*3,2] = -dt*((px_k-cx)*np.sin(w*dt)+(py_k-cy)*np.cos(w*dt))
            #J[i*3,3] = -major*(np.cos(phi_k+w*dt)-np.cos(phi_k))
            #J[i*3,4] =  minor*(np.sin(phi_k+w*dt)-np.sin(phi_k))
            
            #J[i*3+1,0] = 0
            J[i*3+1,1] = dt
            J[i*3+1,2] = dt*(a*np.cos(phi_k+w*dt)-b*np.sin(phi_k+w*dt))
            #J[i*3+1,3] = -major*(np.sin(phi_k+w*dt)-np.sin(phi_k))
            #J[i*3+1,4] = -minor*(np.cos(phi_k+w*dt)-np.cos(phi_k))
    
            J[i*3+2,2] = dt
            
            
        
    else:
        J = np.zeros([3,3])
        phi_k = DF.loc[idx,'theta']
        px_k = DF.loc[idx,'rx']
        py_k =  DF.loc[idx,'ry']
        
             
        J[0,0] = dt
        #J[0,1] = 0
        J[0,2] = -dt*(a*np.sin(phi_k+w*dt)+b*np.cos(phi_k+w*dt))
        #J[0,3] = -major*(np.cos(phi_k+w*dt)-np.cos(phi_k))
        #J[0,4] =  minor*(np.sin(phi_k+w*dt)-np.sin(phi_k))
    
        #J[1,0] = 0
        J[1,1] = dt
        J[1,2] = dt*(a*np.cos(phi_k+w*dt)-b*np.sin(phi_k+w*dt))
        #J[1,3] = -major*(np.sin(phi_k+w*dt)-np.sin(phi_k))
        #J[1,4] = -minor*(np.cos(phi_k+w*dt)-np.cos(phi_k))
    
        J[2,2] = dt
        
    

    return J



#------------3DOF
    
def Res2(vx, vy, gamma1, gamma2, major, minor, dt, DF, idx):  
    if idx=='all':
        px_k = DF['rx']
        px_k2 = DF['rxf']
        py_k =  DF['ry']
        py_k2 = DF['ryf']
        phi_k = DF['theta']
        phi_k2 = DF['thetaf']
        w = DF['w']
        
    else:
        px_k = DF.loc[idx,'rx']
        px_k2 = DF.loc[idx,'rxf']
        py_k =  DF.loc[idx,'ry']
        py_k2 = DF.loc[idx,'ryf']
        phi_k = DF.loc[idx,'theta']
        phi_k2 = DF.loc[idx,'thetaf']
        w = DF.loc[idx, 'w']
    
    
    a = (1-gamma1)*major
    b = (1-gamma2)*minor
    
    px_calc = px_k +a*(np.cos(phi_k+w*dt)-np.cos(phi_k))-b*(np.sin(phi_k+w*dt)-np.sin(phi_k))+vx*dt
    py_calc = py_k +a*(np.sin(phi_k+w*dt)-np.sin(phi_k))+b*(np.cos(phi_k+w*dt)-np.cos(phi_k))+vy*dt
    phi_calc = phi_k+w*dt
    
    R1 = -px_k2 + px_calc
    R2 = -py_k2 + py_calc
    R3 = -phi_k2+ phi_calc
    
    
    #old approach 
    #cx = px_k-a*np.cos(phi_k)+b*np.sin(phi_k) #a*np.cos(phi_k)-b*np.sin(phi_k)
    #cy = py_k-a*np.sin(phi_k)-b*np.cos(phi_k) #a*np.sin(phi_k)+b*np.cos(phi_k)
    #R1 = -px_k2+(px_k-cx)*np.cos(w*dt)-(py_k-cy)*np.sin(w*dt)+cx+vx*dt
    #R2 = -py_k2+(px_k-cx)*np.sin(w*dt)+(py_k-cy)*np.cos(w*dt)+cy+vy*dt
    #R3 = -phi_k2+phi_k+w*dt
    
    res = np.array([R1,R2,R3]).T
    return res



def Res4(vx, vy, gamma1, gamma2, major, minor, dt, DF, idx):  
    if idx=='all':
        px_k = DF['rx']
        px_k2 = DF['rxf']
        py_k =  DF['ry']
        py_k2 = DF['ryf']
        phi_k = DF['theta']
        phi_k2 = DF['thetaf']
        w = DF['w']
        vpx = DF['vpx']
        vpy = DF['vpy']
        
    else:
        px_k = DF.loc[idx,'rx']
        px_k2 = DF.loc[idx,'rxf']
        py_k =  DF.loc[idx,'ry']
        py_k2 = DF.loc[idx,'ryf']
        phi_k = DF.loc[idx,'theta']
        phi_k2 = DF.loc[idx,'thetaf']
        w = DF.loc[idx, 'w']
        vpx = DF.loc[idx, 'vpx']
        vpy = DF.loc[idx, 'vpy']
    
    
    a = (1-gamma1)*major
    b = (1-gamma2)*minor
    
    axbx = a*np.cos(PHI[i])-b*np.sin(PHI[i])
    ayby = a*np.sin(PHI[i])+b*np.cos(PHI[i])
    
    px_calc = px_k +a*(np.cos(phi_k+w*dt)-np.cos(phi_k))-b*(np.sin(phi_k+w*dt)-np.sin(phi_k))+vx*dt
    py_calc = py_k +a*(np.sin(phi_k+w*dt)-np.sin(phi_k))+b*(np.cos(phi_k+w*dt)-np.cos(phi_k))+vy*dt
    phi_calc = phi_k+w*dt
    
    vpx_calc = 1/dt*((np.cos(w*dt)-1)*(axbx)-np.sin(w*dt)*(ayby))
    vpy_calc = 1/dt*(np.sin*(w*dt)*(axbx)-(np.cos(w*dt)-1)*(ayby))
    
    R1 = -px_k2 + px_calc
    R2 = -py_k2 + py_calc
    R3 = -phi_k2+ phi_calc
    R4 = -vpx + vpx_calc
    R5 = -vpy + vpy_calc
    
    
    #old approach 
    #cx = px_k-a*np.cos(phi_k)+b*np.sin(phi_k) #a*np.cos(phi_k)-b*np.sin(phi_k)
    #cy = py_k-a*np.sin(phi_k)-b*np.cos(phi_k) #a*np.sin(phi_k)+b*np.cos(phi_k)
    #R1 = -px_k2+(px_k-cx)*np.cos(w*dt)-(py_k-cy)*np.sin(w*dt)+cx+vx*dt
    #R2 = -py_k2+(px_k-cx)*np.sin(w*dt)+(py_k-cy)*np.cos(w*dt)+cy+vy*dt
    #R3 = -phi_k2+phi_k+w*dt
    
    res = np.array([R1,R2,R3,R4,R5]).T
    return res
