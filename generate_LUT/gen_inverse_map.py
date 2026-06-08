import numpy as np
from scipy.optimize import least_squares

# -----------------------------
# 1) Your forward model
# -----------------------------
def forward_model(VL, VR):
    """
    Return (v, omega) predicted by your calibration equations.
    Replace the body with your actual equations.
    """
    v = -0.26528615 + 0.00364152*VL + 0.00640563*VR - 0.00002286*VL*VL - 0.00005067*VR*VR - 0.00002508*VL*VR \
    + 0.00000010*VL*VL*VL + 0.00000013*VR*VR*VR + 0.00000017*VL*VR*VR - 0.00000005*VL*VL*VR #[m/s]

    omega = -5.088834 + 0.133761*VR - 0.034432*VL - 5.73471732e-04*VR*VR - 6.95066445e-05*VL*VL + 1.00638876e-04*VR*VL #[rad/s]
    return v, omega


# -----------------------------
# 2) Residual function
# -----------------------------
def residuals(x, v_des, omega_des, w_v=1.0, w_omega=1.0):
    VL, VR = x
    v, omega = forward_model(VL, VR)
    return np.array([
        w_v * (v - v_des),
        w_omega * (omega - omega_des),
    ])


# -----------------------------
# 3) Solve for voltages
# -----------------------------
def solve_voltages(v_des, omega_des,
                   x0=(80, 80),
                   Vmin=60, Vmax=120,
                   w_v=1.0, w_omega=1.0):
    res = least_squares(
        residuals,
        x0=np.array(x0, dtype=float),
        bounds=(np.array([Vmin, Vmin]), np.array([Vmax, Vmax])),
        args=(v_des, omega_des, w_v, w_omega),
        method="trf",          # supports bounds
        loss="linear",         # plain least squares (can use 'soft_l1' if noisy)
        ftol=1e-10, xtol=1e-10, gtol=1e-10,
        max_nfev=100
    )
    VL, VR = res.x
    return VL, VR, res

from scipy.optimize import root

def F(x, v_d, w_d):
    VL, VR = x
    v, omega = forward_model(VL, VR)
    return np.array([v - v_d,
                     omega - w_d])

if __name__ == "__main__":

    omega_all = np.load("Calibration/TinyMPC/generate_LUT/omega_table_latest.npy")  #[rad/s]
    velocity_all = np.load("Calibration/TinyMPC/generate_LUT/vCOM_table_latest.npy") #[m/s]

    vmin = np.min(velocity_all)
    vmax = np.max(velocity_all)
    print("velocity range =", vmin, "to", vmax)
    omegamin = np.min(omega_all)
    omegamax = np.max(omega_all)
    print("omega range =", omegamin, "to", omegamax)

    v_des = np.arange(vmin, vmax, 0.001)
    omega_des = np.arange(omegamin, omegamax, 0.01)

    VL_table = np.zeros((len(v_des), len(omega_des)), dtype=float)
    VR_table = np.zeros((len(v_des), len(omega_des)), dtype=float)
    for itr, v in  enumerate(v_des):
        for itr2, omega in enumerate(omega_des):
            print("\nDesired (v, omega) =", v, omega)

            # good practice: use previous solution as x0 in a control loop
            VL, VR, res = solve_voltages(v, omega, x0=(80,80))
            VL_table[itr, itr2] = VL
            VR_table[itr, itr2] = VR

            print("VL, VR =", VL, VR)
            print("cost =", res.cost)   # 0.5 * sum(residuals**2) [web:717]
            print("success =", res.success, res.message)

    print("Lookup table for VL:")
    print(VL_table)
    print("Lookup table for VR:")
    print(VR_table)

    np.savetxt("Calibration/TinyMPC/generate_LUT/VL_table.txt", VL_table, fmt="%.3f", delimiter=",")
    np.savetxt("Calibration/TinyMPC/generate_LUT/VR_table.txt", VR_table, fmt="%.3f", delimiter=",")

    print("omega range =", omegamin, "to", omegamax, "rad/s")
    print("velocity range =", vmin, "to", vmax, "m/s")

    # VL, VR, res = solve_voltages(0.09, -3.46, x0=(80,80))
    # print("VL, VR =", VL, VR)