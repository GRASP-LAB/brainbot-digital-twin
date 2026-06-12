import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
from pathlib import Path

base = Path(__file__).resolve().parent

sns.set_theme(style="white")
cmap = sns.color_palette("mako", as_cmap=True)

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.titlesize": 16,
    "lines.linewidth": 1.0, 
})

fig, ax = plt.subplots(figsize=(12, 5), nrows=1, ncols=2)

### Velocity model: 3rd-order bivariate polynomial
# ── 1. Load data ─────────────────────────────────────────────────────────────
velocity_table = np.load(base / "vCOM_table_latest.npy")  # shape (7, 7)
print(np.round(velocity_table, 4))
power = np.load(base / "motorpower_table_latest.npy")     # shape (7,)

### To verify the meshgrid construction, print the grids, and notice that (V_L,V_R) = (60,80) corresponds to almost straight motion.
### i.e., left motor is stronger than the right motor.

# rows = left motor power, cols = right motor power
power_L_grid, power_R_grid = np.meshgrid(power, power, indexing="ij")
power_L_grid = power_L_grid.T

power_R_grid = np.rot90(power_R_grid)

pL_flat  = power_L_grid.ravel().astype(float)
pR_flat  = power_R_grid.ravel().astype(float)
vel_flat = velocity_table.ravel()*100  # convert to cm/s for better visualization

# ── 2. Model ──────────────────────────────────────────────────────────────────
def velocity_model(X, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9):
    """Degree-3 bivariate polynomial in (pL, pR)."""
    pL, pR = X
    return (c0
            + c1 * pL + c2 * pR
            + c3 * pL**2 + c4 * pR**2
            + c5 * pL * pR
            + c6 * pL**3 + c7 * pR**3
            + c8 * pR**2 * pL + c9 * pL**2 * pR
            )

# ── 3. Fit ────────────────────────────────────────────────────────────────────
p0 = np.zeros(10)
popt, pcov = curve_fit(velocity_model, (pL_flat, pR_flat), vel_flat, p0=p0)
perr = np.sqrt(np.diag(pcov))

c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = popt
print(popt)

print("=" * 55)
print("Fitted coefficients (velocity = f(pL, pR)):")
print(f"  c0 (bias)     = {c0:+.6f}  ± {perr[0]:.2e}")
print(f"  c1 (pL)       = {c1:+.6f}  ± {perr[1]:.2e}")
print(f"  c2 (pR)       = {c2:+.6f}  ± {perr[2]:.2e}")
print(f"  c3 (pL^2)     = {c3:+.6f}  ± {perr[3]:.2e}")
print(f"  c4 (pR^2)     = {c4:+.6f}  ± {perr[4]:.2e}")
print(f"  c5 (pL*pR)    = {c5:+.6f}  ± {perr[5]:.2e}")
print(f"  c6 (pL^3)     = {c6:+.6f}  ± {perr[6]:.2e}")
print(f"  c7 (pR^3)     = {c7:+.6f}  ± {perr[7]:.2e}")
print(f"  c8 (pR^2*pL)  = {c8:+.6f}  ± {perr[8]:.2e}")
print(f"  c9 (pL^2*pR)  = {c9:+.6f}  ± {perr[9]:.2e}")
print("=" * 55)

# ── 4. Goodness-of-fit ────────────────────────────────────────────────────────
vel_pred  = velocity_model((pL_flat, pR_flat), *popt)
residuals = vel_flat - vel_pred
ss_res    = np.sum(residuals**2)
ss_tot    = np.sum((vel_flat - vel_flat.mean())**2)
R2        = 1.0 - ss_res / ss_tot
RMSE      = np.sqrt(np.mean(residuals**2))

print(f"\nGoodness-of-fit:")
print(f"  R²   = {R2:.6f}")
print(f"  RMSE = {RMSE:.6f} cm/s")
print("=" * 55)

pL_fine = np.linspace(power.min(), power.max(), 60)
pR_fine = np.linspace(power.min(), power.max(), 60)
PL, PR  = np.meshgrid(pL_fine, pR_fine)
VEL_fit = velocity_model((PL, PR), *popt).reshape(PL.shape)

# --- 5. Visualize -----------------------------
levels=np.linspace(4.8, 11.5, 51)
cf = ax[0].contourf(PL, PR, VEL_fit, levels=levels, cmap=cmap, extend="both")
cbar = plt.colorbar(cf, ax=ax[0], format='%.1f')
cbar.set_label(r"$v\,(cm/s)$")

# optional: overlay original sampled grid points
ax[0].scatter(pL_flat, pR_flat, c=vel_flat, cmap=cmap, edgecolor="k", s=30, vmin=4.8, vmax=11.5)

ax[0].set_xlim(power.min()-2, power.max()+2)
ax[0].set_ylim(power.min()-2, power.max()+2)

ax[0].set_xlabel(r"$V_\mathrm{L}$")
ax[0].set_ylabel(r"$V_\mathrm{R}$")
# ax[0].set_title("Calibration for velocity (cm/s)")


# #### Angular velocity fit: 2nd-order bivariate polynomial
# # ── 1. Load data ────────────────────────────────────────────────────────────
omega_table = np.load(base / "omega_table_latest.npy")   # shape (7, 7)
power       = np.load(base / "motorpower_table_latest.npy")         # shape (7,)  e.g. [60..120]

### To verify the meshgrid construction, print the grids, and notice that (V_L,V_R) = (60,60) corresponds to clockwise motion (-ve omega).
### i.e., left motor is stronger than the right motor.

## Build meshgrid: rows = left motor power, cols = right motor power
power_L_grid, power_R_grid = np.meshgrid(power, power, indexing="ij")

power_L_grid = power_L_grid.T
power_R_grid = np.rot90(power_R_grid)

# Flatten to 1-D arrays for curve_fit
pL_flat    = power_L_grid.ravel().astype(float)
pR_flat    = power_R_grid.ravel().astype(float)
omega_flat = omega_table.ravel()

# ── 2. Model definition ─────────────────────────────────────────────────────
def omega_model(X, c0, c1, c2, c3, c4, c5):
    """Degree-2 bivariate polynomial in (pL, pR)."""
    pL, pR = X
    return (c0
            + c1 * pL + c2 * pR
            + c3 * pL**2 + c4 * pR**2
            + c5 * pL * pR)

# ── 3. Fit ───────────────────────────────────────────────────────────────────
p0 = np.zeros(6)   # initial guess
popt, pcov = curve_fit(omega_model, (pL_flat, pR_flat), omega_flat, p0=p0)
perr = np.sqrt(np.diag(pcov))
print(popt)
c0, c1, c2, c3, c4, c5 = popt

print("=" * 55)
print("Fitted coefficients (omega = f(pL, pR)):")
print(f"  c0 (bias)     = {c0:+.6f}  ± {perr[0]:.2e}")
print(f"  c1 (pL)       = {c1:+.6f}  ± {perr[1]:.2e}")
print(f"  c2 (pR)       = {c2:+.6f}  ± {perr[2]:.2e}")
print(f"  c3 (pL^2)     = {c3:+.6f}  ± {perr[3]:.2e}")
print(f"  c4 (pR^2)     = {c4:+.6f}  ± {perr[4]:.2e}")
print(f"  c5 (pL*pR)    = {c5:+.6f}  ± {perr[5]:.2e}")
print("=" * 55)

# ── 4. Residuals & goodness-of-fit ──────────────────────────────────────────
omega_pred = omega_model((pL_flat, pR_flat), *popt)
residuals  = omega_flat - omega_pred
ss_res     = np.sum(residuals**2)
ss_tot     = np.sum((omega_flat - omega_flat.mean())**2)
R2         = 1.0 - ss_res / ss_tot
RMSE       = np.sqrt(np.mean(residuals**2))

print(f"\nGoodness-of-fit:")
print(f"  R²   = {R2:.6f}")
print(f"  RMSE = {RMSE:.6f} rad/s")
print("=" * 55)

#### ──  Visualise ─────────────────────────────────────────────────────────────

pL_fine = np.linspace(power.min(), power.max(), 60)
pR_fine = np.linspace(power.min(), power.max(), 60)
PL, PR  = np.meshgrid(pL_fine, pR_fine)
OMEGA_fit = omega_model((PL, PR), *popt).reshape(PL.shape)

levels = np.linspace(-3.46, 1.43, 51)
cf = ax[1].contourf(PL, PR, OMEGA_fit, levels=levels, cmap=cmap, extend="both")
cbar = plt.colorbar(cf, ax=ax[1], format='%.2f')
cbar.set_label(r"$\omega\,(rad/s)$")

# optional: overlay original sampled grid points
ax[1].scatter(pL_flat, pR_flat, c=omega_flat, cmap=cmap, edgecolor="k", s=30, vmin=-3.46, vmax=1.43)

ax[1].set_xlim(power.min()-2, power.max()+2)
ax[1].set_ylim(power.min()-2, power.max()+2)

ax[1].set_xlabel(r"$V_\mathrm{L}$")
ax[1].set_ylabel(r"$V_\mathrm{R}$")

fig.text(0.01, 0.95, r'(a)', fontsize=16)
fig.text(0.5, 0.95, r'(b)', fontsize=16)
fig.tight_layout()
fig.savefig(base / "Calibration_maps.pdf", dpi=600)
plt.show()

