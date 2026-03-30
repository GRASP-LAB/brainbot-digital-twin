import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import count
import random




rad = lambda x: x*np.pi/180
deg = lambda x: x*180/np.pi
Rot = lambda angle_rad: np.array([[np.cos(angle_rad), -np.sin(angle_rad)],[np.sin(angle_rad), np.cos(angle_rad)]])

def rotate_translate(X,Y, h,k, phi): #rotates X,Y point by phi radians and translate h,k respectively
    X_rt = X*np.cos(phi)-Y*np.sin(phi)+h
    Y_rt = X*np.sin(phi)+Y*np.cos(phi)+k
    
    return X_rt, Y_rt


class Ell2D():
    
    id_iter = count()
    
    def __init__(self, r=np.array([0,0]), theta_rad=rad(0), v=np.array([0,0]), w=0, a=1, b=0.5, noise=0, npoints=101, ctend=[1,1]): 
        #head is self.body[0,:]
        self.ID = next(self.id_iter)
  
        #dimensions
        self.a = a;
        self.b = b;
        self.mass=1

        #velocity
        self.v=v
        self.w=w
        self.acc = 0
        self.rot_acc=0
        self.noise = noise
        
        self.v0 = self.v
        self.w0 = self.w
        
        #position 
        self.r = r # position (x,y)=(r[0], r[1])
              
        #direction
        t_size = npoints #number of points to plot the ellipsoids
        self.npoints = npoints
        t = np.linspace(0, 2*np.pi, t_size)
        self.theta = theta_rad
        if abs(self.theta)>2*np.pi:
            self.theta=np.sign(self.theta)*(abs(self.theta)%(2*np.pi))
            
               
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

        
        #center of rotations as weighted averages, but given with simplified correlation.
        #first edge points
        ##self.q1 = self.r-self.a*self.n1 
        ##self.q2 = self.r-self.b*self.n2
        #now center of rot
        ##self.c1 = self.r*ctend[0]+self.q1*(1-ctend[0])
        ##self.c2 = self.r*ctend[1]+self.q2*(1-ctend[1])
        #self.c1 = self.r-(1-self.ctend[0])*self.a*self.n1
        #self.c2 = self.r-(1-self.ctend[1])*self.b*self.n2
        #self.com = self.c1+self.c2-self.r
        
        
        #Inertia #RECALCULATE THIS WITH NEW COM.
        self.inertia = False #self.mass*(0.25*(self.a**2+self.b**2))
        
        #Following two lines rotates the initial ellipse around center regardless of com
        self.body=self.body@Rot(self.theta).T   #rotation
        self.body+=r                            #translation
        self.c1 = self.r-(1-self.ctend[0])*self.a*self.n1
        self.c2 = self.r-(1-self.ctend[1])*self.b*self.n2
        self.com = self.c1+self.c2-self.r

        #self.n0 = self.n.copy()
        
    #end class=======================================================================    
    
    def on_ellipse(self):
        return (1-self.ctend[0])**2+(1-self.ctend[1])**2<=1
    
    def reset_vel(self):
        self.v  = self.v0
        #self.n  = self.n0
        self.w  = self.w0
        
    
    
    def translate(self, dr):
        self.body = self.body+dr
        self.r = self.r+dr
        self.c1 = self.c1+dr
        self.c2 = self.c2+dr
        self.com = self.com+dr
        
    
    
    def rotate(self, dtheta):
        self.body = (self.body-self.com)@Rot(dtheta).T+self.com
        self.r = Rot(dtheta)@(self.r-self.com)+self.com
        self.c1 = Rot(dtheta)@(self.c1-self.com)+self.com
        self.c2 = Rot(dtheta)@(self.c2-self.com)+self.com
        
        self.com = self.c1+self.c2-self.r
        self.n1 = Rot(dtheta)@self.n1
        self.n2 = Rot(dtheta)@self.n2
        
        self.theta = self.theta+dtheta
        #if abs(self.theta)>2*np.pi:
            #self.theta=sign(self.theta)*(abs(self.theta)%(2*np.pi))
        
        
        
           
        
    def move(self,dt):
        dr = dt*self.v
        dtheta = dt*self.w

        self.rotate(dtheta)
        self.translate(dr)
        
        
    
    #NEW
    def calc_alpha(self, force_rot=0, force_arm=0):
        inertia_com = self.inertia+self.mass*(force_arm)**2
        Rot_Acc = (force_arm*np.linalg.norm(force_rot))/inertia_com #?? is it right formula especially norm(force_rot)
        return Rot_Acc
    
    def updateVel(self, dt, force_tr, alpha):
        #update velocity
        v_new = self.v*self.n+force_tr*dt
        self.v=np.linalg.norm(v_new)
        if self.v<=1e-10:
            self.n = np.array([0,0])
            self.v = 0
        else:
            self.n = v_new/self.v
            
        #update angular velocity
        self.w = self.w+alpha*dt
    
    
    def update_velocity(self, dt, force_tr=0, force_rot=0, force_arm=0):
        v_new = self.v*self.n+force_tr*dt
        self.v = np.linalg.norm(v_new)
        if self.v<=1e-10:
            self.n = np.array([0,0])
            self.v = 0
            
        else:
            self.n = v_new/self.v
           
         
        inertia_com = self.inertia+self.mass*(force_arm)**2 #inertia as per parallel axis theorem
        rotational_accelaration = (force_arm*np.linalg.norm(force_rot))/inertia_com
        
        w_new = self.w+rotational_accelaration*dt
        #print("w = {}\tw_new = {}\trot_acc*dt ={}\t".format(self.w, w_new, rotational_accelaration*dt))
        self.w = w_new
    
    def cardinal_points(self):
        step = int(round((self.npoints-1)/4))
        return 0,2*step,3*step,4*step
        
    def data(self, message=True):
        if message:
            print('r,com,theta,v,w,ctend')
        return self.r, self.com, self.theta, self.v, self.w, self.ctend
        
    def vec(self, unit=True):
        vi= self.r-self.com
        if unit:
            vi = vi/np.linalg.norm(vi)
        return vi
        
        
        
        
    