import numpy as np
import scipy.optimize
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
#from EllipseObj import *
import matplotlib.pyplot as plt



def filter_XY(df,dtime, filtering, window, pol_degree):
    win=window
    
    if filtering==False:
        #pol_degree=6
        X = df.X
        Y = df.Y
        vpxf = df.vx
        vpyf = df.vy
        wf  = df.w
        Thetaf = df.Theta
        
        return X,Y,Thetaf, vpxf,vpyf, wf
    elif filtering==True: 
        #pol_degree=6
        X = savgol_filter(df.X, win,pol_degree, 0)
        Y = savgol_filter(df.Y, win,pol_degree, 0)
        vpxf = savgol_filter(df.X, win,pol_degree, 1)/dtime
        vpyf = savgol_filter(df.Y, win,pol_degree, 1)/dtime
        wf = savgol_filter(df.Theta, win,pol_degree, 1)/dtime
        Thetaf = savgol_filter(df.Theta, win,pol_degree, 0)
        
        return X,Y,Thetaf, vpxf,vpyf, wf
        
    elif "vel_only":
        X = df.X
        Y = df.Y
        Thetaf = df.Theta
        vpxf = savgol_filter(df.X, win,pol_degree, 1)/dtime
        vpyf = savgol_filter(df.Y, win,pol_degree, 1)/dtime
        wf = savgol_filter(df.Theta, win,pol_degree, 1)/dtime
        
        return X,Y,Thetaf, vpxf,vpyf, wf
        


def derive_ICR(df,dtime, filtering, window=31, pol_degree=6):

    X,Y,Theta, vpxf,vpyf, wf = filter_XY(df,dtime, filtering, window, pol_degree)
    icrx = X-vpyf/wf
    icry = Y+vpxf/wf
    icr = np.array([icrx,icry]).T
    
    return icr




def fitTrajectory(df, pol_degree):
    l = len(df.X)
    X = df.X[:l]
    Y = df.Y[:l]
    Theta = df.Theta[:l]

    X0 = X[0]
    Y0 = Y[0]
    Theta0 = Theta[0]
    t = df.Time[:l]

    num_params_poly = pol_degree + 1
    total_params = 2 * num_params_poly + 2

    def model_function(t, *params):
        a_coeffs = params[:num_params_poly]
        b_coeffs = params[num_params_poly:2 * num_params_poly]
        gA, gB = params[2 * num_params_poly:]

        x_poly = np.polyval(a_coeffs[::-1], t)
        y_poly = np.polyval(b_coeffs[::-1], t)

        x_sin_cos = gA * (np.cos(Theta) - np.cos(Theta0)) - gB * (np.sin(Theta) - np.sin(Theta0))
        y_sin_cos = gA * (np.sin(Theta) - np.sin(Theta0)) + gB * (np.cos(Theta) - np.cos(Theta0))

        return np.hstack((x_poly + x_sin_cos, y_poly + y_sin_cos))

    Z = np.hstack((X, Y))
    initial_guess = np.ones(total_params)
    params, covariance = curve_fit(model_function, t, Z, p0=initial_guess)

    def PolSinFit(t):
        a_coeffs = params[:num_params_poly]
        b_coeffs = params[num_params_poly:2 * num_params_poly]
        gA, gB = params[2 * num_params_poly:]

        x_poly = np.polyval(a_coeffs[::-1], t)
        y_poly = np.polyval(b_coeffs[::-1], t)

        x_sin_cos = gA * (np.cos(df.Theta[:len(t)]) - np.cos(Theta0)) - gB * (np.sin(df.Theta[:len(t)]) - np.sin(Theta0))
        y_sin_cos = gA * (np.sin(df.Theta[:len(t)]) - np.sin(Theta0)) + gB * (np.cos(df.Theta[:len(t)]) - np.cos(Theta0))

        x_total = x_poly + x_sin_cos
        y_total = y_poly + y_sin_cos

        return x_total, y_total

    def LinFit(t):
        a_coeffs = params[:num_params_poly]
        b_coeffs = params[num_params_poly:2*num_params_poly]
        return np.polyval(a_coeffs[::-1], t), np.polyval(b_coeffs[::-1], t)

    def SinFit(t):
        gA,gB= params[2*num_params_poly:]
        x_sin_cos = gA * (np.cos(df.Theta[:len(t)]) - np.cos(Theta0)) - gB * (np.sin(df.Theta[:len(t)]) - np.sin(Theta0))
        y_sin_cos = gA * (np.sin(df.Theta[:len(t)]) - np.sin(Theta0)) + gB * (np.cos(df.Theta[:len(t)]) - np.cos(Theta0))
        return x_sin_cos, y_sin_cos

    return params, PolSinFit, LinFit, SinFit


#simplified code, for the Spiral model. Any polynomial degree can be used and the intercept (coefficients of 0 degree terms) are taken as X0,Y0
def fitTrajectory2(df, pol_degree):
    l = len(df.X)
    X = df.X[:l]
    Y = df.Y[:l]
    Theta = df.Theta[:l]

    X0 = X[0]
    Y0 = Y[0]
    Theta0 = Theta[0]
    t = df.Time[:l]

    num_params_poly = pol_degree  # Reduced by 1 since we are fixing the constant term
    total_params = 2 * num_params_poly + 2

    def model_function(t, *params):
        a_coeffs = np.hstack(([X0], params[:num_params_poly]))
        b_coeffs = np.hstack(([Y0], params[num_params_poly:2 * num_params_poly]))
        gA, gB = params[2 * num_params_poly:]

        x_poly = np.polyval(a_coeffs[::-1], t)
        y_poly = np.polyval(b_coeffs[::-1], t)

        x_sin_cos = gA * (np.cos(Theta) - np.cos(Theta0)) - gB * (np.sin(Theta) - np.sin(Theta0))
        y_sin_cos = gA * (np.sin(Theta) - np.sin(Theta0)) + gB * (np.cos(Theta) - np.cos(Theta0))

        return np.hstack((x_poly + x_sin_cos, y_poly + y_sin_cos))

    Z = np.hstack((X, Y))
    initial_guess = np.ones(total_params)
    params, covariance = curve_fit(model_function, t, Z, p0=initial_guess)

    def PolSinFit(t):
        a_coeffs = np.hstack(([X0], params[:num_params_poly]))
        b_coeffs = np.hstack(([Y0], params[num_params_poly:2 * num_params_poly]))
        gA, gB = params[2 * num_params_poly:]

        x_poly = np.polyval(a_coeffs[::-1], t)
        y_poly = np.polyval(b_coeffs[::-1], t)

        x_sin_cos = gA * (np.cos(df.Theta[:len(t)]) - np.cos(Theta0)) - gB * (np.sin(df.Theta[:len(t)]) - np.sin(Theta0))
        y_sin_cos = gA * (np.sin(df.Theta[:len(t)]) - np.sin(Theta0)) + gB * (np.cos(df.Theta[:len(t)]) - np.cos(Theta0))

        x_total = x_poly + x_sin_cos
        y_total = y_poly + y_sin_cos

        return x_total, y_total

    def LinFit(t):
        a_coeffs = np.hstack(([X0], params[:num_params_poly]))
        b_coeffs = np.hstack(([Y0], params[num_params_poly:2 * num_params_poly]))
        return np.polyval(a_coeffs[::-1], t), np.polyval(b_coeffs[::-1], t)

    def SinFit(t):
        gA, gB = params[2 * num_params_poly:]
        x_sin_cos = gA * (np.cos(df.Theta[:len(t)]) - np.cos(Theta0)) - gB * (np.sin(df.Theta[:len(t)]) - np.sin(Theta0))
        y_sin_cos = gA * (np.sin(df.Theta[:len(t)]) - np.sin(Theta0)) + gB * (np.cos(df.Theta[:len(t)]) - np.cos(Theta0))
        return x_sin_cos, y_sin_cos

    return params, PolSinFit, LinFit, SinFit


#simplified code, for the Spiral model. Only 1st degree polynomial is assumed and the intercept (coefficients of 0 degree terms) are taken as X0,Y0
def fitTrajectory_spiral1(df, initial_guess = np.ones(4)):# equivalent to fitTrajectory2(df, 1)
    X = df.X
    Y = df.Y
    Theta = df.Theta

    X0 = X[0]
    Y0 = Y[0]
    Theta0 = Theta[0]
    t = df.Time

    total_params = 4  # 2 for the linear terms and 2 for the sinusoidal terms

    def model_function(t, a1, b1, gA, gB):
        x_poly = X0 + a1 * t
        y_poly = Y0 + b1 * t

        x_sin_cos = gA * (np.cos(Theta) - np.cos(Theta0)) - gB * (np.sin(Theta) - np.sin(Theta0))
        y_sin_cos = gA * (np.sin(Theta) - np.sin(Theta0)) + gB * (np.cos(Theta) - np.cos(Theta0))

        return np.hstack((x_poly + x_sin_cos, y_poly + y_sin_cos))

    Z = np.hstack((X, Y))
    
    params, _ = curve_fit(model_function, t, Z, p0=initial_guess)

    def PolSinFit(t):
        a1, b1, gA, gB = params

        x_poly = X0 + a1 * t
        y_poly = Y0 + b1 * t

        x_sin_cos = gA * (np.cos(df.Theta[:len(t)]) - np.cos(Theta0)) - gB * (np.sin(df.Theta[:len(t)]) - np.sin(Theta0))
        y_sin_cos = gA * (np.sin(df.Theta[:len(t)]) - np.sin(Theta0)) + gB * (np.cos(df.Theta[:len(t)]) - np.cos(Theta0))

        return x_poly + x_sin_cos, y_poly + y_sin_cos

    def LinFit(t):
        a1, b1, _, _ = params
        return X0 + a1 * t, Y0 + b1 * t

    def SinFit(t):
        _, _, gA, gB = params
        x_sin_cos = gA * (np.cos(df.Theta[:len(t)]) - np.cos(Theta0)) - gB * (np.sin(df.Theta[:len(t)]) - np.sin(Theta0))
        y_sin_cos = gA * (np.sin(df.Theta[:len(t)]) - np.sin(Theta0)) + gB * (np.cos(df.Theta[:len(t)]) - np.cos(Theta0))
        return x_sin_cos, y_sin_cos

    return params, PolSinFit, LinFit, SinFit
    
    

def fitSin(df, pol_degree):
    l = len(df.X)
    X = df.X[:l]
    Y = df.Y[:l]
    Theta = df.Theta[:l]

    X0 = X[0]
    Y0 = Y[0]
    Theta0 = Theta[0]
    t = df.Time[:l]

    #num_params_poly = pol_degree + 1
    total_params = 4

    def model_function(t, *params):
        a_coeffs = params[0]
        b_coeffs = params[1]
        gA, gB = params[2:]

        x_sin_cos = a_coeffs+gA * (np.cos(Theta) - np.cos(Theta0)) - gB * (np.sin(Theta) - np.sin(Theta0))
        y_sin_cos = b_coeffs+gA * (np.sin(Theta) - np.sin(Theta0)) + gB * (np.cos(Theta) - np.cos(Theta0))

        return np.hstack((x_sin_cos, y_sin_cos))

    Z = np.hstack((X, Y))
    initial_guess = np.ones(total_params)
    params, covariance = curve_fit(model_function, t, Z, p0=initial_guess)


    def SinFit(t):
        gA,gB= params[2:]
        x_sin_cos = params[0]+gA * (np.cos(df.Theta[:len(t)]) - np.cos(Theta0)) - gB * (np.sin(df.Theta[:len(t)]) - np.sin(Theta0))
        y_sin_cos = params[1]+gA * (np.sin(df.Theta[:len(t)]) - np.sin(Theta0)) + gB * (np.cos(df.Theta[:len(t)]) - np.cos(Theta0))
        return x_sin_cos, y_sin_cos
        
    def SinFit(t):
        gA,gB= params[2:]
        x_sin_cos = params[0]+gA * (np.cos(df.Theta[:len(t)]) - np.cos(Theta0)) - gB * (np.sin(df.Theta[:len(t)]) - np.sin(Theta0))
        y_sin_cos = params[1]+gA * (np.sin(df.Theta[:len(t)]) - np.sin(Theta0)) + gB * (np.cos(df.Theta[:len(t)]) - np.cos(Theta0))
        return x_sin_cos, y_sin_cos

    return params, SinFit


def fitTrajectoryWindowed(df, pol_degree, win_size):
    LEN = len(df)
    n = int(LEN/win_size)+1
    Xps,Yps,Xl,Yl, Xs, Ys = [],[],[],[],[],[]
    for i in range(n):
        llim = i*win_size
        rlim = min(LEN, (i+1)*win_size)
        if (llim<rlim):
            params, psf, lf, sf = fitTrajectory(df[llim:rlim].reset_index(), pol_degree)

            xps,yps = psf(df.Time[llim:rlim])
            xlin,ylin = lf(df.Time[llim:rlim])
            xs,ys = sf(df.Time[llim:rlim])

            Xps.extend(xps); Yps.extend(yps)
            Xl.extend(xlin); Yl.extend(ylin)
            Xs.extend(xs); Ys.extend(ys)

    return np.vstack((Xps, Yps,Xl,Yl, Xs, Ys))


def fitSinWindowed(df, pol_degree, win_size):
    LEN = len(df)
    n = int(LEN/win_size)+1
    Xs, Ys = [],[]
    ax,ay=[],[]
    for i in range(n):
        llim = i*win_size
        rlim = min(LEN, (i+1)*win_size)
        if (llim<rlim):
            params, sf = fitSin(df[llim:rlim].reset_index(), pol_degree)
            ax.append(params[0])
            ay.append(params[1])
            xs,ys = sf(df.Time[llim:rlim])

            Xs.extend(xs); Ys.extend(ys)

    return np.vstack((ax, ay)), np.vstack((Xs, Ys))
    
    
    
    
def cor_parameters(df,dtime, filtering=True, window=31, pol_degree=6, compute_vc=True):
    win=window
    if filtering: 
        #pol_degree=6
        X = savgol_filter(df.X, win,pol_degree, 0)
        Y = savgol_filter(df.Y, win,pol_degree, 0)
        vpxf = savgol_filter(df.X, win,pol_degree, 1)/dtime
        vpyf = savgol_filter(df.Y, win,pol_degree, 1)/dtime
        wf = savgol_filter(df.Theta, win,pol_degree, 1)/dtime
        Thetaf = savgol_filter(df.Theta, win,pol_degree, 0)
    elif "vel_only":
        X = df.X
        Y = df.Y
        Thetaf = df.Theta
        vpxf = savgol_filter(df.X, win,pol_degree, 1)/dtime
        vpyf = savgol_filter(df.Y, win,pol_degree, 1)/dtime
        wf = savgol_filter(df.Theta, win,pol_degree, 1)/dtime
    else:
        
        #pol_degree=6
        X = df.X
        Y = df.Y
        vpxf = df.vx
        vpyf = df.vy
        wf  = df.w
        Thetaf = df.Theta
        
    icrx = X-vpyf/wf
    icry = Y+vpxf/wf
    icr = np.array([icrx,icry]).T
    if compute_vc:
        vcx = np.gradient(icrx)/dtime
        vcy = np.gradient(icry)/dtime
        vc = np.array([vcx,vcy]).T
        
        
        corx = X-(vpyf-vcy)/wf
        cory = Y+(vpxf-vcx)/wf
        cor = np.array([corx,cory]).T
        
        rx = (vpyf-vcy)/wf
        ry = -(vpxf-vcx)/wf
        r = np.array([rx,ry]).T
    else:
        rx = (vpyf)/wf
        ry = -(vpxf)/wf
        r = np.array([rx,ry]).T
        vc = np.zeros([0,0])
        cor = np.zeros([0,0])
    g = np.zeros((r.shape[0],2))
    glist = []
    
    for i in range(r.shape[0]):
        Rinv = Rot(Thetaf[i]).T
        ab = Rinv@r[i,:]
        #print(r.shape)
        g[i,0] = 1-ab[0]/MAJOR
        g[i,1] = 1-ab[1]/MINOR
        glist.append(list(g[i,:]))
    
    
    return icr, vc, cor, g,glist
 



cf=0.0006742675003064854
#Constant
MAJOR = 81.57/2*cf
MINOR = 44.542/2*cf 
    
    
def plot_coms(g1,g2):
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
    #plt.scatter(c[0], c[1], c='blue', label='Average (in Ellipse)')
