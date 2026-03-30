import numpy as np
import matplotlib.pyplot as plt


def lambda_fft(dt, t, w, threshold, fmin=None, fmax=None, plotting=True):
    w_ave = 0  # or use np.average(w) if needed
    v = w - w_ave
    N = len(w)
    
    f = np.fft.fftfreq(N, dt)
    V = np.fft.fft(v)
    amp = np.abs(V) / N
    amp_unfiltered = amp.copy()
    phs = np.arctan2(np.imag(V), np.real(V))

    # --- FIXED filtering logic ---
    mask = amp >= threshold
    if fmin is not None:
        mask &= np.abs(f) >= fmin
    if fmax is not None:
        mask &= np.abs(f) <= fmax
    amp[~mask] = 0
    # -----------------------------
    print('Amp[mask] = ',amp[mask].shape)

    if plotting:
        f_pos = f[f >= 0]
        amp_pos = amp_unfiltered[f >= 0]
        
        plt.figure(figsize=(10, 4))
        plt.plot(f_pos[1:], amp_pos[1:], label='FFT magnitude')
        plt.axhline(threshold, color='r', linestyle='--', label='Threshold')
        plt.axvline(fmin, color='r', linestyle='-.', label='Threshold')
        plt.axvline(fmax, color='r', linestyle='--', label='Threshold')

        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.title('Magnitude Spectrum with Threshold')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    vrec_func = lambda t_input: np.sum([
        amp[k] * np.cos(2 * np.pi * f[k] * t_input + phs[k]) for k in range(N)
    ], axis=0)

    return vrec_func

def erf0(sim_params, theta_interpolator,w_interpolator, direction=1, v_interpolate=False):
    vo, vh_1, vh_2 = sim_params
    sgn_vo = direction

    if v_interpolate:
        fv1 = lambda t: (vo(t) + vh_1(t) * np.cos(theta_interpolator(t)) - vh_2(t) * np.sin(theta_interpolator(t)))
        fv2 = lambda t: (sgn_vo * vo(t) - vh_2(t) * np.cos(theta_interpolator(t)) - vh_1(t) * np.sin(theta_interpolator(t)))
    else:
        fv1 = lambda t: (         vo + vh_1 * np.cos(theta_interpolator(t)) - vh_2 * np.sin(theta_interpolator(t)))
        fv2 = lambda t: (sgn_vo * vo - vh_2 * np.cos(theta_interpolator(t)) - vh_1 * np.sin(theta_interpolator(t)))
    return fv1, fv2

def erf1(sim_params, theta_interpolator, w_interpolator, direction=1, v_interpolate=False):
    vo_1, vo_2, vh_1, vh_2 = sim_params

    if v_interpolate:
        fv1 = lambda t: (vo_1(t) + vh_1(t) * np.cos(theta_interpolator(t)) - vh_2(t) * np.sin(theta_interpolator(t)))
        fv2 = lambda t: (vo_2(t) - vh_1(t) * np.cos(theta_interpolator(t)) - vh_2(t) * np.sin(theta_interpolator(t)))
    else:
        fv1 = lambda t: (vo_1 + vh_1 * np.cos(theta_interpolator(t)) - vh_2 * np.sin(theta_interpolator(t)))
        fv2 = lambda t: (vo_2 - vh_1 * np.cos(theta_interpolator(t)) - vh_2 * np.sin(theta_interpolator(t)))
    return fv1, fv2

def erf2(sim_params, theta_interpolator,w_interpolator, direction=1, v_interpolate=False):
    vo, vh_1, vh_2 = sim_params
    sgn_vo = direction

    if v_interpolate:
        fv1 = lambda t: (         vo(t) + vh_1(t) * np.cos(vh_2(t) + theta_interpolator(t)))
        fv2 = lambda t: (sgn_vo * vo(t) - vh_1(t) * np.sin(vh_2(t) + theta_interpolator(t)))
    else:
        fv1 = lambda t: (         vo + vh_1 * np.cos(vh_2 + theta_interpolator(t)))
        fv2 = lambda t: (sgn_vo * vo - vh_1 * np.sin(vh_2 + theta_interpolator(t)))
    return fv1, fv2

def erf3(sim_params, theta_interpolator,w_interpolator, direction=1, v_interpolate=False):
    vo, vh_1 = sim_params
    sgn_vo = direction

    if v_interpolate:
        fv1 = lambda t: (         vo(t) + vh_1(t) * np.cos(theta_interpolator(t)))
        fv2 = lambda t: (sgn_vo * vo(t) - vh_1(t) * np.sin(theta_interpolator(t)))
    else:
        fv1 = lambda t: (         vo + vh_1 * np.cos(theta_interpolator(t)))
        fv2 = lambda t: (sgn_vo * vo - vh_1 * np.sin(theta_interpolator(t)))
    return fv1, fv2

def erf4(sim_params, theta_interpolator,w_interpolator, direction=1, v_interpolate=False):
    vo_1, vo_2, vh_1 = sim_params

    if v_interpolate:
        fv1 = lambda t: (vo_1(t) + vh_1(t) * np.cos(theta_interpolator(t)))
        fv2 = lambda t: (vo_2(t) - vh_1(t) * np.sin(theta_interpolator(t)))
    else:
        fv1 = lambda t: (vo_1 + vh_1 * np.cos(theta_interpolator(t)))
        fv2 = lambda t: (vo_2 - vh_1 * np.sin(theta_interpolator(t)))
    return fv1, fv2

def erf5(sim_params, theta_interpolator,w_interpolator, direction=1, v_interpolate=False):
    vo_1, vo_2, = sim_params
    if v_interpolate:
        fv1 = lambda t: (vo_1(t))
        fv2 = lambda t: (vo_2(t))
    else:
        fv1 = lambda t: (vo_1)
        fv2 = lambda t: (vo_2)
    return fv1, fv2

def erf6(sim_params, theta_interpolator,w_interpolator, direction=1, v_interpolate=False):
    vo_1, vo_2, vh_1, vh_2 = sim_params
    sgn_vo = direction

    if v_interpolate:
        fv1 = lambda t: (vo_1(t) + vh_1(t) * np.cos(vh_2(t) + theta_interpolator(t)))
        fv2 = lambda t: (vo_2(t) - vh_1(t) * np.sin(vh_2(t) + theta_interpolator(t)))
    else:
        fv1 = lambda t: (vo_1 + vh_1 * np.cos(vh_2 + theta_interpolator(t)))
        fv2 = lambda t: (vo_2 - vh_1 * np.sin(vh_2 + theta_interpolator(t)))
    return fv1, fv2

def erf7(sim_params, theta_interpolator,w_interpolator, direction=1, v_interpolate=False):
    vo_1, vo_2, vh_1, vh_2, vh_3 = sim_params
    sgn_vo = direction

    if v_interpolate:
        fv1 = lambda t: (vo_1(t) + vh_1(t) * np.cos(vh_3(t) + theta_interpolator(t)))
        fv2 = lambda t: (vo_2(t) - vh_2(t) * np.sin(vh_3(t) + theta_interpolator(t)))
    else:
        fv1 = lambda t: (vo_1 + vh_1 * np.cos(vh_3 + theta_interpolator(t)))
        fv2 = lambda t: (vo_2 - vh_2 * np.sin(vh_3 + theta_interpolator(t)))
    return fv1, fv2

def erf8(sim_params, theta_interpolator,w_interpolator, direction=1, v_interpolate=False, w_div = False):
    vo_1, vo_2, vh_1, vh_2, vh_3 = sim_params
    sgn_vo = direction
    w = lambda t: np.abs(w_interpolator(t))
    #vh_3 = np.abs(vh_3)%(2*np.pi)*np.sign(vh_3)
    if v_interpolate:
        fv1 = lambda t: 1/w(t)*(vo_1(t) + vh_1(t) * np.cos(vh_3(t) + theta_interpolator(t)))
        fv2 = lambda t: 1/w(t)*(vo_2(t) - vh_2(t) * np.sin(vh_3(t) + theta_interpolator(t)))
    else:
        fv1 = lambda t: 1/w(t)*(vo_1 + vh_1 * np.cos(vh_3 + theta_interpolator(t)))
        fv2 = lambda t: 1/w(t)*(vo_2 - vh_2 * np.sin(vh_3 + theta_interpolator(t)))
    return fv1, fv2

def erf9(sim_params, theta_interpolator, w_interpolator, direction=1, v_interpolate=False):
    vo_1, vo_2, vh_1, vh_2 = sim_params
    w = w_interpolator
    if v_interpolate:
        fv1 = lambda t: 1/w(t)*(vo_1(t) + vh_1(t) * np.cos(theta_interpolator(t)) - vh_2(t) * np.sin(theta_interpolator(t)))
        fv2 = lambda t: 1/w(t)*(vo_2(t) + vh_2(t) * np.cos(theta_interpolator(t)) + vh_1(t) * np.sin(theta_interpolator(t)))
    else:
        fv1 = lambda t: 1/w(t)*(vo_1 + vh_1 * np.cos(theta_interpolator(t)) - vh_2 * np.sin(theta_interpolator(t)))
        fv2 = lambda t: 1/w(t)*(vo_2 + vh_2 * np.cos(theta_interpolator(t)) + vh_1 * np.sin(theta_interpolator(t)))
    return fv1, fv2

def erf10(sim_params, theta_interpolator,w_interpolator, direction=1, v_interpolate=False, w_div = False):
    vo_1, vo_2, vh_1, vh_2, vh_3, vh_4 = sim_params
    sgn_vo = direction
    w = lambda t: np.abs(w_interpolator(t))
    #vh_3 = np.abs(vh_3)%(2*np.pi)*np.sign(vh_3)
    if v_interpolate:
        fv1 = lambda t: 1/w(t)*(vo_1(t) + vh_1(t) * np.cos(vh_3(t) + theta_interpolator(t)))
        fv2 = lambda t: 1/w(t)*(vo_2(t) - vh_2(t) * np.sin(vh_4(t) + theta_interpolator(t)))
    else:
        fv1 = lambda t: 1/w(t)*(vo_1 + vh_1 * np.cos(vh_3 + theta_interpolator(t)))
        fv2 = lambda t: 1/w(t)*(vo_2 - vh_2 * np.sin(vh_4 + theta_interpolator(t)))
    return fv1, fv2

def erf11(sim_params, theta_interpolator, w_interpolator, direction=1, v_interpolate=False):
    vo_1, vo_2, vh_1, vh_2, vh_3, vh_4 = sim_params
    w = lambda t: np.abs(w_interpolator(t))
    if v_interpolate:
        fv1 = lambda t: (vo_1(t) + vh_1(t) * np.cos(theta_interpolator(t)) - vh_2(t) * np.sin(theta_interpolator(t)))
        fv2 = lambda t: (vo_2(t) - vh_3(t) * np.cos(theta_interpolator(t)) - vh_4(t) * np.sin(theta_interpolator(t)))
    else:
        fv1 = lambda t: (vo_1 + vh_1 * np.cos(theta_interpolator(t)) - vh_2 * np.sin(theta_interpolator(t)))
        fv2 = lambda t: (vo_2 - vh_3 * np.cos(theta_interpolator(t)) - vh_4 * np.sin(theta_interpolator(t)))
    return fv1, fv2

def erf12(sim_params, theta_interpolator, w_interpolator, direction=1, v_interpolate=False):
    vo_1, vo_2, vh_1, vh_2, vh_3, vh_4 = sim_params
    w = lambda t: np.abs(w_interpolator(t))
    if v_interpolate:
        fv1 = lambda t: (1/w(t))*(vo_1(t) + vh_1(t) * np.cos(theta_interpolator(t)) - vh_2(t) * np.sin(theta_interpolator(t)))
        fv2 = lambda t: (1/w(t))*(vo_2(t) - vh_3(t) * np.cos(theta_interpolator(t)) - vh_4(t) * np.sin(theta_interpolator(t)))
    else:
        fv1 = lambda t: (1/w(t))*(vo_1 + vh_1 * np.cos(theta_interpolator(t)) - vh_2 * np.sin(theta_interpolator(t)))
        fv2 = lambda t: (1/w(t))*(vo_2 - vh_3 * np.cos(theta_interpolator(t)) - vh_4 * np.sin(theta_interpolator(t)))
    return fv1, fv2

def erf13(sim_params, theta_interpolator, w_interpolator, direction=1, v_interpolate=False):
    vo_1, vo_2, b = sim_params
    w = lambda t: np.abs(w_interpolator(t))
    if v_interpolate:
        fv1 = lambda t: (1/w(t))**b*(vo_1(t))
        fv2 = lambda t: (1/w(t))**b*(vo_2(t))
    else:
        fv1 = lambda t: (1/w(t))**b*(vo_1)
        fv2 = lambda t: (1/w(t))**b*(vo_2)
    return fv1, fv2


erfs = [erf0, erf1, erf2, erf3, erf4, erf5, erf6, erf7,erf8, erf9, erf10,erf11,erf12,erf13]
erf_param_len = [3,       4,    3,    2,    3,    2,    4,    5,   5,    6,     6,    6,    6, 3]


