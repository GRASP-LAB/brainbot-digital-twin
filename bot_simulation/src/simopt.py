import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from itertools import count

plt.rcParams['font.size'] = 16
plt.figure(figsize=(12,12))

rad = lambda x: x*np.pi/180
deg = lambda x: x*180/np.pi
Rot = lambda angle_rad: np.array([[np.cos(angle_rad), -np.sin(angle_rad)],[np.sin(angle_rad), np.cos(angle_rad)]])
cf=0.0006742675003064854
#Constant
MAJOR = 5.5/2/100#81.57/2*cf/
MINOR = 3.0/2/100#44.542/2*cf
Xmax=792*cf
Ymax=718*cf

rho1 = 0.374
rho2 = 0.661
rmag = np.sqrt((rho1*MAJOR)**2+(rho2*MINOR)**2)

norm = np.linalg.norm
class Ell2D():
    id_iter = count()
    def __init__(self, 
                 r = np.array([0,0]), 
                 theta_rad = rad(0), 
                 t0=0,
                 npoints = 101,
                 vfunc1=np.sin, vfunc2=np.sin, wfunc=np.sin, cm_scale=False): 
        
        #head is self.body[0,:]
        #self.ID = next(Ell.id_iter)
        #dimensions
        self.a = 5.5*1e-2/2; #m
        self.b = 3.0*1e-2/2; #m
        if cm_scale:  
            self.a*=100
            self.b*=100            
        #self.a = 81.57/2*cf
        #self.b = 44.542/2*cf
        self.data = []

        #position 
        self.r = r # position (x,y)=(r[0], r[1])
        
        #set orientations
        self.theta = theta_rad
        self.n1, self.n2 = self.calculate_normals()
        
        self.mass=1
        self.Rctend = np.array([1-rho1,1-rho2])#np.array([0.6,0.4]) CW rotation
        self.Lctend = np.array([1-rho1,1+rho2])#np.array([0.6,1.6]) CCW rotation
        self.rmag = norm((1-self.Rctend)@np.array([MAJOR,MINOR]))
        #check if it is correct   
        self.inertia = 0.25*(self.a**2+self.b**2)+((1-self.Rctend[0])*self.a)**2+((1-self.Rctend[1])*self.b)**2
        self.drot = np.linalg.norm((self.Rctend-self.Lctend)*np.array([self.a, self.b]))
        
        #velocity
        self.v=np.array([0.,0.])
        self.w=0.
        self.acc = 0.
        self.rot_acc=0.
        self.noise = 0.
        self.time = t0
        self.time_0 = t0
        self.vfunc1 = vfunc1
        self.vfunc2 = vfunc2
        self.wfunc = wfunc
        
        #direction
        #number of points to plot the ellipsoids
        self.npoints = npoints
        t = np.linspace(0, 2*np.pi, npoints)
        
        if abs(self.theta)>2*np.pi:
            self.theta=np.sign(self.theta)*(abs(self.theta)%(2*np.pi))
           
        #orientation        
        self.ellBody=np.zeros([self.npoints,2])
        self.ellBody[:,0] = self.a*np.cos(t)
        self.ellBody[:,1] = self.b*np.sin(t)
        #initialize body
        self.body = self.ellBody.copy()
        
        #direction vectors
        self.calculate_normals()
        self.reconstructBody()
        
        self.ctend = np.array([0.,0.])
        self.set_ctend()
        self.com = self.calculate_com(self.ctend)
        self.theta_previous = self.theta
        #self.gatherData()
            
            
    #com is position calculated with ctend and geometric center
    def set_ctend(self):
        if self.w<=0: #CW, 
            self.ctend = self.Rctend.copy()
        elif self.w>0: #CCW rotation
            self.ctend = self.Lctend.copy()
    
    def calculate_com(self, ctend): 
        com = np.zeros(2)
        com[0] = self.r[0]-(1-ctend[0])*self.a*np.cos(self.theta)+(1-ctend[1])*self.b*np.sin(self.theta)
        com[1] = self.r[1]-(1-ctend[0])*self.a*np.sin(self.theta)-(1-ctend[1])*self.b*np.cos(self.theta)
        return com  
    
    def reconstructBody(self):
        self.body=self.ellBody@Rot(self.theta).T   #rotation
        self.body+=self.r  #translation        

    def calculate_normals(self):
        n1 = np.array([np.cos(self.theta), np.sin(self.theta)])
        n2 = np.array([-np.sin(self.theta), np.cos(self.theta)])
        return n1, n2

    def updateNormals(self):
        self.n1 = np.array([np.cos(self.theta), np.sin(self.theta)])
        self.n2 = np.array([-np.sin(self.theta), np.cos(self.theta)])
    
    def calculateLocalVels(self):
        n1 = np.array([np.cos(self.theta), np.sin(self.theta)])
        n2 = np.array([-np.sin(self.theta), np.cos(self.theta)])
        return self.v@n1, self.v@n2

    def plot_body(self):
        plt.plot(self.body[:,0], self.body[:,1])
        plt.scatter(self.com[0], self.com[1])
        plt.quiver(self.r[0], self.r[1], self.n1[0], self.n1[1])
        
    def updateVelocities(self, dt):
        #print(Fm)        
        n1, n2 = self.calculate_normals()
        self.v = self.vfunc1(self.time)*n1+self.vfunc2(self.time)*n2
        self.w = self.wfunc(self.time)
        self.set_ctend()
        self.com = self.calculate_com(self.ctend)

       
    def move(self, dt):
        self.time+=dt
        #update Positions
        self.theta_previous = self.theta
        self.theta += self.w*dt
        self.r = Rot(self.w*dt)@(self.r-self.com)+self.com+self.v*dt
        self.com = self.com + self.v*dt
        #update direction
        self.updateNormals()
        #update Body as well
        self.reconstructBody()
        #update Velocities
        self.updateVelocities(dt)
        #print(self.ctend)
        self.gatherData()
    
    def gatherData(self):
        v1,v2 = self.calculateLocalVels()
        self.data.append([self.time, self.r[0], self.r[1], self.theta,self.com[0], self.com[1], self.v[0], self.v[1], v1,v2, norm(self.v), self.w])
        
    def getData(self):
        return pd.DataFrame(self.data, columns='time,X,Y,Theta,cx,cy,vx,vy,v1,v2,v,w'.split(','))

    def simulate(self, time, dt, icr='geom'):
        nt = int(np.round(time/dt))
        for i in range(nt):
            self.move(dt)
        df = self.getData()

        rx = (1-self.ctend[0])*MAJOR*np.cos(df.Theta)-(1-self.ctend[1])*MINOR*np.sin(df.Theta)
        ry = (1-self.ctend[0])*MAJOR*np.sin(df.Theta)+(1-self.ctend[1])*MINOR*np.cos(df.Theta)
        #df['vpx'] = np.gradient(df.X)/dt
        #df['vpy'] = np.gradient(df.Y)/dt
        
        df['vpx'] = df.vx-df.w*ry
        df['vpy'] = df.vy+df.w*rx
        df['vpmag'] = np.sqrt(df.vpx**2+df.vpy**2)
        df['cos'] = np.cos(df.Theta)
        df['sin'] = np.sin(df.Theta)

        #df.vpx=df.vpx.shift(-1)
        
        df['sigma'] = np.abs(df.w*rmag)/df.vpmag
        
        df['Gicry'] = df.Y+df.vpx/df.w
        df['Gicrx'] = df.X-df.vpy/df.w
        
        #df['Ticry'] = df.cy+df.vx/df.w
        #df['Ticrx'] = df.cx-df.vy/df.w  

        return df 

    def plot_spin_circle(self, direction):
        
        if direction.upper()=='CW':
            print('doing cw: ', self.Rctend)
            com = self.calculate_com(self.Rctend)
        elif direction.upper()=='CCW':
            print('doing ccw: ', self.Lctend )
            com = self.calculate_com(self.Lctend)
        
        Rv = self.r-com
        R = np.sqrt(Rv@Rv)
        t = np.linspace(0,2*np.pi, 100)
        x = com[0]+R*np.cos(t)
        y = com[1]+R*np.sin(t)
        return x,y
#----------------------------------------------------------------------------
class BBot():
    id_iter = count()
    def __init__(self, 
                 r = np.array([0,0]), 
                 theta_rad = rad(0), 
                 t0=0,
                 npoints = 101,
                 vfunc1=np.sin, vfunc2=np.sin, wfunc=np.sin, cm_scale=False): 
        
        #head is self.body[0,:]
        #self.ID = next(Ell.id_iter)
        #dimensions
        self.A1 = 5.5*1e-2/2; #m
        self.A2 = 3.0*1e-2/2; #m
        if cm_scale:  
            self.A1*=100
            self.A2*=100     

        
        self.set_rho(wfunc(0))
        self.rmag = np.sqrt((self.rho1*self.A1)**2+(self.rho2*self.A2)**2)
        

        self.data = []
        #position 
        self.r = r # position (x,y)=(r[0], r[1])
        #set orientations
        self.theta = theta_rad
        self.n1 = np.array([1,0]) #dummy values
        self.n2 = np.array([0,1]) #dummy values
        self.n1f = lambda phi: np.array([ np.cos(phi), np.sin(phi)])
        self.n2f = lambda phi: np.array([-np.sin(phi), np.cos(phi)])
        #velocity
        self.v=np.array([0.,0.])
        self.w=0.

        self.time = t0
        self.time_0 = t0
        self.vf1 = vfunc1
        self.vf2 = vfunc2
        self.wf  = wfunc

        #direction
        if abs(self.theta)>2*np.pi:
            self.theta=np.sign(self.theta)*(abs(self.theta)%(2*np.pi))
           
        #direction vectors
        self.updateNormals()
        #self.reconstructBody()
        
        self.ctend = np.array([0.,0.])
        #self.set_rho()
        self.rc = self.calculate_rc()
        self.theta_previous = self.theta
        #self.gatherData()
        self.df = pd.DataFrame()

    #com is position calculated with ctend and geometric center
    def set_rho(self, w):
        if w<=0: #CW, 
            self.rho1 = -0.374
            self.rho2 = -0.661
        elif w>0: #CCW rotation
            self.rho1 = -0.374
            self.rho2 =  0.661
    
    def calculate_rc(self): 
        rc = np.zeros(2)
        rc = self.r+self.rho1*self.A1*self.n1+self.rho2*self.A2*self.n2
        return rc  

    def updateNormals(self):
        self.n1 = self.n1f(self.theta)
        self.n2 = self.n2f(self.theta)
    
    def gatherData(self):
        v1 = self.v@self.n1
        v2 = self.v@self.n2
        self.data.append([self.time, self.r[0], self.r[1], self.theta,self.rc[0], self.rc[1], self.v[0], self.v[1], v1,v2, norm(self.v), self.w, self.rho1,self.rho2])
        
    def getData(self):
        self.df =  pd.DataFrame(self.data, columns='time,X,Y,Theta,cx,cy,vx,vy,v1,v2,v,w,rho1,rho2'.split(','))
        return self.df
    def simulate(self, time, dt, icr='geom'):
        nt = int(np.round(time/dt))
        for i in range(nt):
            self.move(dt)
        df = self.getData()
        self.df = df

    def postprocess(self):
        df = self.df
        rho2 = self.rho2
        rho1 = self.rho1
        A1   = self.A1
        A2   = self.A2
        rmrcx = -(rho1*A1*np.cos(df.Theta)-rho2*A2*np.sin(df.Theta)) #(r-rc)_x
        rmrcy = -(rho1*A1*np.sin(df.Theta)+rho2*A2*np.cos(df.Theta))

        df['vpx'] = df.vx - df.w*rmrcy
        df['vpy'] = df.vy + df.w*rmrcx
        df['vpmag'] = np.sqrt(df.vpx**2+df.vpy**2)

        df['vpx_grad'] = np.gradient(df.X)/np.gradient(df.time)
        df['vpy_grad'] = np.gradient(df.Y)/np.gradient(df.time)
        
        df['sigma'] = np.abs(df.w*self.rmag)/df.vpmag
        df['Gicrx'] = df.X-df.vpy/df.w
        df['Gicry'] = df.Y+df.vpx/df.w

        if np.all((df.rho2*df.w)<0):
            print("w and rho2 dont align.")

        self.df = df 



def simulate_bbot(Bot, time, dt, integrator='ExpEu', dependency='time', w_correction=False):
    if integrator=='ExpEu':
        return integrate_ExpEu(Bot, time, dt)
    elif integrator=='CN':
        return integrate_CN(Bot, time, dt, dependency, w_correction)

def integrate_ExpEu(Bot, time, dt):
    nt = int(np.round(time/dt))
    B = Bot
    for i in range(nt):
        B.time += dt
        t = B.time
        B.set_rho(B.w) #since this is ExpEu velocities from prev time used

        B.theta_previous = B.theta
        B.theta += B.w*dt
        B.updateNormals()

        B.r  = Rot(B.w*dt)@(B.r-B.rc)+ B.rc + B.v*dt
        B.rc = B.rc + B.v*dt
        #B.rc = B.rc + B.v*dt
        #B.r  = B.rc - (rho1*A1*B.n1+rho2*A2*B.n2)

        #Bot.updateVelocities(dt)
        B.v = B.vf1(t)*B.n1+B.vf2(t)*B.n2
        B.w = B.wf(t)
        #B.rc = B.calculate_rc()
        B.gatherData()
    B.getData()
    B.postprocess()
    df = B.df
    return df

def integrate_CN(Bot, time, dt, dependency, w_correction):
    nt = int(np.round(time/dt))
    B = Bot
    for i in range(nt):
        t0 = B.time
        t1 = t0+dt
        B.time = t1

        w = 0.5*(B.wf(t0)+B.wf(t1))
        B.set_rho(w)
        th0 = B.theta
        th1 = B.theta+w*dt
        B.theta = th1
        B.updateNormals()
        

        #B.theta_previous = B.theta
        #B.theta += w*dt
        if dependency == 'time':
            v_t0 = B.vf1(t0)*B.n1f(th0)+B.vf2(t0)*B.n2f(th0) #with theta_previous
            v_t1 = B.vf1(t1)*B.n1f(th1)+B.vf2(t1)*B.n2f(th1) #with theta
        elif dependency == 'angle':
            v_t0 = B.vf1(th0)*B.n1f(th0)+B.vf2(th0)*B.n2f(th0) #with theta_previous
            v_t1 = B.vf1(th1)*B.n1f(th1)+B.vf2(th1)*B.n2f(th1) #with theta

        if w_correction: #unnecessary, can be removed.
            v_t0 = v_t0/np.abs(w)
            v_t1 = v_t1/np.abs(w)

        v = 0.5*(v_t0+v_t1)


        B.r  = Rot(w*dt)@(B.r-B.rc) + B.rc + v*dt
        B.rc = B.rc + v*dt

        B.w = B.wf(t1) 
        B.v = v_t1 

        #Bot.updateVelocities(dt)
        #B.rc = B.calculate_rc()
        B.gatherData()
    B.getData()
    B.postprocess()
    df = B.df
    return df

#--------------------------------------------------------------------------

def plot_sim(df, i_step=0, plot_cor=True):
    fig = plt.figure(figsize=(18, 10))
    # Create a GridSpec with a 3x3 layout
    gs = fig.add_gridspec(3, 3)
    # First plot: takes a 2x2 space
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax1.plot(df.X, df.Y); plt.xlabel('X,(m)'); plt.ylabel('Y,(m)')
    plt.scatter(df.X[0], df.Y[0])
    ax1.set_title('Trajectory')

    #b0 = Ell2D(r = np.array([df.X[i_step],  df.Y[i_step]]), theta_rad = df.Theta[i_step], vfunc1=fv, vfunc2=fv2, wfunc=fw)
    b0 = Ell2D(r = np.array([df.X[i_step],  df.Y[i_step]]), theta_rad = df.Theta[i_step])
    body = b0.body.copy()
    if plot_cor:
        plt.plot(df.Gicrx[1:len(df)-1],df.Gicry[1:len(df)-1], label='Gicr'); plt.scatter(df.Gicrx[0],df.Gicry[0])
    #plt.plot(df.Ticrx[1:len(df)-1],df.Ticry[1:len(df)-1], label='Ticr'); plt.scatter(df.Ticrx[0],df.Ticry[0])
    #plt.plot(df.Ticrx,df.Ticry, label='Ticry'); plt.scatter(df.Ticrx[0],df.Ticry[0])
    plt.quiver(df.X[i_step],  df.Y[i_step], np.cos(df.Theta[i_step]), np.sin(df.Theta[i_step]))
    plt.quiver(df.X[i_step],  df.Y[i_step], -np.sin(df.Theta[i_step]), np.cos(df.Theta[i_step]))
    plt.quiver(df.X[i_step],  df.Y[i_step], np.cos(np.pi/3), np.sin(np.pi/3), color='red')

    plt.plot(body[:,0], body[:,1], c='grey')
    plt.axis('square')

    
    # Second plot
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(df.time, df.v1, label='v1')
    ax2.plot(df.time, df.v2, label='v2')
    plt.legend(); plt.xlabel('time,(s)')

    # Third plot
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.plot(df.time, df.w)

    plt.xlabel('time,(s)'); plt.ylabel('w,(rad/s)')
    
    # Fourth plot
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(df.time, df.X, label='X')
    ax4.plot(df.time, df.Y, label='Y')
    plt.legend(); plt.xlabel('time,(s)')
    
   
    
    # Fifth plot
    ax5 = fig.add_subplot(gs[2, 1])
    #ax5.set_title('Plot 5')
    ax5.plot(df.time, df.vpx, label='vpx')
    ax5.plot(df.time, df.vpy, label='vpy')
    plt.legend(); plt.xlabel('time,(s)')
    plt.ylabel('vp,(m/s)')

    
    
    # Sixth plot
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.plot(df.time, df.sigma)
    plt.xlabel('time,(s)'); plt.ylabel('sigma')
    plt.ylim([0,2])
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()  


def plot_coms(g1,g2, color='grey'):
    b = Ell(theta_rad=rad(90), r=np.array([0.2, 0.2]), noise=0, v=0, w=0,  a=MAJOR, b=MINOR, ctend=0, npoints=100)
    #b = Ell2D(r=np.array([0, 0]),theta_rad=np.pi/2, a=MAJOR, b=MINOR)
    plt.plot(b.body[:,0], b.body[:,1], lw=3.5, c='black')
    

    c_ave = np.array([0,0])
    for i in range(len(g1)):
        c1 = b.r-(1-g1[i])*MAJOR*np.array([np.cos(rad(90)), np.sin(rad(90))])
        c2 = b.r-(1-g2[i])*MINOR*np.array([-np.sin(rad(90)), np.cos(rad(90))])
        c = c1+c2-b.r
        c_ave=c_ave+c
        plt.scatter(c[0], c[1], label='G-{}'.format(i))
    c_ave = c_ave/len(g1)   
    plt.ylim([0.17,0.23]); plt.xlim([0.17,0.23])
    plt.legend()

def ellipse_points(g1,g2):
    t=np.linspace(0,np.pi*2, 100)
    bx = MINOR*(np.cos(t))
    by = MAJOR*(np.sin(t))
    cx = (1-g1)*MAJOR
    #gA = lambda g1: (1-g1)*MAJOR
    #gB = lambda g2: (1-g2)*MINOR
    #plt.scatter(gB(1.6), -gA(0.6))
    cx = (1-g2)*MINOR
    cy =-(1-g1)*MAJOR
    return bx,by, cx, cy 


def computeVC(vpx,vpy, w, Theta, ga,gb):
    rx = (1-ga)*MAJOR*np.cos(Theta)-(1-gb)*MINOR*np.sin(Theta)
    ry = (1-ga)*MAJOR*np.sin(Theta)+(1-gb)*MINOR*np.cos(Theta)
    
    vcx = vpx+w*ry
    vcy = vpy-w*rx
    
    vc1 = np.multiply(vcx,np.cos(Theta)) + np.multiply(vcy, np.sin(Theta))
    vc2 = -np.multiply(vcx,np.sin(Theta)) + np.multiply(vcy, np.cos(Theta))
    return vcx,vcy,vc1,vc2
    
    
def plot_bot_legs(g1,g2,l=0.08, lines=True, ):  
    bx,by,_,_ = ellipse_points(0,0)
    plt.plot(bx,by, 'black')
    Gg =    [[1.490553554,0.503333333],
         [1.261462645,0.503333333],
         [1.032371736,0.503333333],
         [0.803280827, 0.503333333],
         [0.574189918, 0.503333333],
         [1.490553554,1.496666667],
         [1.261462645,1.496666667],
         [1.032371736,1.496666667],
         [0.803280827, 1.496666667],
         [0.574189918, 1.496666667]]
    def Circle(c,r, npoints=100):
        t=np.linspace(0,np.pi*2, npoints)
        x = c[0]+r*np.cos(t)
        y = c[1]+r*np.sin(t)
        return x,y
    for G in  Gg:
        _,_,cx,cy = ellipse_points(G[0],G[1])
        x,y = Circle([cx,cy], 0.00270/2)
        plt.plot(x,y, 'black')
    if lines:
        x_arr=np.linspace(0.009009966635535589,1,6)
        cyfR = lambda m: -0.011000001738780659+m*(x_arr-0.009009966635535589)
        plt.plot(x_arr, cyfR(-1), ls='--', c='gray')
        plt.plot(x_arr, cyfR(0),ls='--', c='gray')
        plt.plot(x_arr, cyfR(-10005),ls='--', c='gray')
        
        x_arrL=np.linspace(-1,-0.009009966899595443,6)
        cyfL = lambda m:  -0.011000000000000003+m*(x_arrL+0.009009966635535589)
        #plt.plot(x_arr[0]*np.ones(6), cyf(1))
        plt.plot(x_arrL, cyfL(0), ls='--', c='gray')
        plt.plot(x_arrL, cyfL(1), ls='--', c='gray')
        plt.plot(x_arrL, cyfL(10005), ls='--', c='gray')
        plt.xlim([-l,l]); plt.ylim([-l,l])
    for i in  range(len(g1)):
        _,_,cx,cy = ellipse_points(g1[i],g2[i])
        plt.scatter(cx,cy, c='red')


def plot_bot_legs_cm(g1,g2,l=8, lines=True, edgeclr='blue'):  
    bx,by,_,_ = ellipse_points(0,0)
    plt.plot(bx*100,by*100, 'black')
    Gg =    [[1.490553554,0.503333333],
         [1.261462645,0.503333333],
         [1.032371736,0.503333333],
         [0.803280827, 0.503333333],
         [0.574189918, 0.503333333],
         [1.490553554,1.496666667],
         [1.261462645,1.496666667],
         [1.032371736,1.496666667],
         [0.803280827, 1.496666667],
         [0.574189918, 1.496666667]]
    def Circle(c,r, npoints=100):
        t=np.linspace(0,np.pi*2, npoints)
        x = c[0]+r*np.cos(t)
        y = c[1]+r*np.sin(t)
        return x,y
    for G in  Gg:
        _,_,cx,cy = ellipse_points(G[0],G[1])
        x,y = Circle([cx,cy], 0.00270/2)
        plt.plot(x*100,y*100, 'black')
    if lines:
        x_arr=np.linspace(0.009009966635535589,1,6)
        cyfR = lambda m: -0.011000001738780659+m*(x_arr-0.009009966635535589)
        plt.plot(x_arr*100, cyfR(-1)*100, ls='--', c='gray')
        plt.plot(x_arr*100, cyfR(0)*100,ls='--', c='gray')
        plt.plot(x_arr*100, cyfR(-10005)*100,ls='--', c='gray')
        
        x_arrL=np.linspace(-1,-0.009009966899595443,6)
        cyfL = lambda m:  -0.011000000000000003+m*(x_arrL+0.009009966635535589)
        #plt.plot(x_arr[0]*np.ones(6), cyf(1))
        plt.plot(x_arrL*100, cyfL(0)*100, ls='--', c='gray')
        plt.plot(x_arrL*100, cyfL(1)*100, ls='--', c='gray')
        plt.plot(x_arrL*100, cyfL(10005)*100, ls='--', c='gray')
        plt.xlim([-l,l]); plt.ylim([-l,l])
    for i in  range(len(g1)):
        _,_,cx,cy = ellipse_points(g1[i],g2[i])
        plt.scatter(cx*100,cy*100, c='red', edgecolor=edgeclr)

def plot_bot_legs_dynamic(g1, g2, base_l=0.03, lines=True):
    bx, by, _, _ = ellipse_points(0, 0)
    plt.plot(bx, by, 'black')
    
    Gg = [[1.490553554, 0.503333333],
          [1.261462645, 0.503333333],
          [1.032371736, 0.503333333],
          [0.803280827, 0.503333333],
          [0.574189918, 0.503333333],
          [1.490553554, 1.496666667],
          [1.261462645, 1.496666667],
          [1.032371736, 1.496666667],
          [0.803280827, 1.496666667],
          [0.574189918, 1.496666667]]
    
    def Circle(c, r, npoints=100):
        t = np.linspace(0, np.pi * 2, npoints)
        x = c[0] + r * np.cos(t)
        y = c[1] + r * np.sin(t)
        return x, y
    
    max_l = base_l  # Start with the base limit
    
    # Plot the reference ellipses
    for G in Gg:
        _, _, cx, cy = ellipse_points(G[0], G[1])
        x, y = Circle([cx, cy], 0.00270 / 2)
        plt.plot(x, y, 'black')
    
    # Plot the lines if enabled
    if lines:
        x_arr = np.linspace(0.009009966635535589, 1, 6)
        cyfR = lambda m: -0.011000001738780659 + m * (x_arr - 0.009009966635535589)
        plt.plot(x_arr, cyfR(-1), ls='--', c='gray')
        plt.plot(x_arr, cyfR(0), ls='--', c='gray')
        plt.plot(x_arr, cyfR(-10005), ls='--', c='gray')
        
        x_arrL = np.linspace(-1, -0.009009966899595443, 6)
        cyfL = lambda m: -0.011000000000000003 + m * (x_arrL + 0.009009966635535589)
        plt.plot(x_arrL, cyfL(0), ls='--', c='gray')
        plt.plot(x_arrL, cyfL(1), ls='--', c='gray')
        plt.plot(x_arrL, cyfL(10005), ls='--', c='gray')
    
    # Plot the points and dynamically adjust the plot limits
    for i in range(len(g1)):
        _, _, cx, cy = ellipse_points(g1[i], g2[i])
        plt.scatter(cx, cy, c='red')
        
        # Update max_l if necessary
        max_l = max(max_l, abs(cx), abs(cy))
    
    # Set dynamic limits based on the largest cx, cy
    plt.xlim([-max_l, max_l])
    plt.ylim([-max_l, max_l])

# Now, calling plot_bot_legs with g1 and g2 will dynamically adjust the plot limits
 


def calculate_G_simulated(data):
    data['Time']=data.time
    params, PolSinFit, LinFit, SinFit = fitTrajectory2(data, 1)
    psx,psy = PolSinFit(data.Time)
    lsx,lsy = LinFit(data.Time)
    sx,sy = SinFit(data.Time)
    #plt.plot(psx,psy)
    #plt.plot(lsx,lsy)
    ga = params[-2]
    gb = params[-1]
    #gs1.append(1-ga/MAJOR)
    GS1 = 1-ga/MAJOR
    #gs2.append(1-gb/MINOR)
    GS2 = 1-gb/MINOR
    return [GS1,GS2]


def plotBot(h,k,phi, npts=100, scale='m'):
    t=np.linspace(0,2*np.pi,npts)
    x = MAJOR*np.cos(t)*np.cos(phi)-MINOR*np.sin(t)*np.sin(phi)+h
    y = MAJOR*np.cos(t)*np.sin(phi)+MINOR*np.sin(t)*np.cos(phi)+k
    if scale=='cm':
        h*=100
        k*=100
        x*=100
        y*=100
    plt.quiver(h,k, np.cos(phi), np.sin(phi))
    plt.plot(x,y, c='black', lw=3)

def ArrayBot(h,k,phi, npts=100, scale='m'):
    t=np.linspace(0,2*np.pi,npts)
    x = MAJOR*np.cos(t)*np.cos(phi)-MINOR*np.sin(t)*np.sin(phi)+h
    y = MAJOR*np.cos(t)*np.sin(phi)+MINOR*np.sin(t)*np.cos(phi)+k
    if scale=='cm':
        h*=100
        k*=100
        x*=100
        y*=100
    return x,y