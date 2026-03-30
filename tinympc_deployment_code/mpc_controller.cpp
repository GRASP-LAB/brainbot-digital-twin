/*
 * mpc_controller.cpp
 * 
 * Summary:
 * This is the actual implementation of the TinyMpcController. It sets up the mathematical
 * problem for the TinyMPC solver to figure out how the robot should move.
 * 
 * Flow:
 * - init(): Pre-computes the circle LUT (look-up table of reference states).
 * - setState(): Saves the robot's current position and speed as the starting point.
 * - solve(): Finds the target spot on the circle based on time, builds an Xref matrix
 *   (the reference trajectory broadcast across the 10-step horizon), calls tiny_solve,
 *   and extracts the first optimal acceleration [ax, ay] from solution->u column 0.
 * - toUnicycle(): Takes the acceleration, integrates one step to get target velocity,
 *   and calculates forward and turning speeds (v and w) the robot actually understands.
 * - _buildAndUpdate(): Helper that fills Xref and Uref and pushes them to the solver.
 */

#include "mpc_controller.h"

#include <math.h>
#include <string.h>


//  Helper: clamp a float to [lo, hi]
static inline float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

//  Helper: wrap angle to [-pi, pi]
static inline float wrap_angle(float a) {
    while (a >  (float)M_PI) a -= 2.0f * (float)M_PI;
    while (a < -(float)M_PI) a += 2.0f * (float)M_PI;
    return a;
}

// Constructor
TinyMpcController::TinyMpcController() : _last_status(-1) {
    memset(_x0, 0, sizeof(_x0));
}


//  init() — call once in setup()
void TinyMpcController::init() {
    // (LUT computed externally in setup)
}

//  setState() — update the current robot state before solve()
void TinyMpcController::setState(float px, float py, float vx, float vy) {
    _x0[0] = (tinytype)px;
    _x0[1] = (tinytype)py;
    _x0[2] = (tinytype)vx;
    _x0[3] = (tinytype)vy;
}

//  solve() — one MPC iteration
bool TinyMpcController::solve(const MpcRefState& ref, float& ax_out, float& ay_out) {
    // Build and push Xref/Uref, set x0
    _buildAndUpdate(ref);

    // Solve
    int status = tiny_solve(&tiny_solver);
    _last_status = status;

    if (status != 0) {
        ax_out = 0.0f;
        ay_out = 0.0f;
        return false;
    }

    // Extract first optimal control u0 = [ax, ay] from solution->u (nu x N-1)
    // Column 0 = first time step
    ax_out = (float)tiny_solver.solution->u(0, 0);
    ay_out = (float)tiny_solver.solution->u(1, 0);

    return true;
}

//  toUnicycle() — feedback-linearisation: [ax,ay] -> [v_cmd, w_cmd]
//
//  Strategy
//    1. Integrate ax,ay over one dt to get target world velocity
//    2. Compute body speed magnitude -> v_cmd
//    3. Compute heading error -> w_cmd = delta_theta / dt
void TinyMpcController::toUnicycle(float ax, float ay, float yaw,
                                    float& v_cmd, float& w_cmd) const {
    const float dt = MPC_DT;  // same Ts used in MPC

    // Target world velocity after one step
    float vx_tgt = (float)_x0[2] + ax * dt;
    float vy_tgt = (float)_x0[3] + ay * dt;

    
    // Target heading and angular rate
    float theta_tgt   = atan2f(vy_tgt, vx_tgt);
    float delta_theta = wrap_angle(theta_tgt - yaw);
    w_cmd = delta_theta / dt;

    float cr;
    if (w_cmd<0) {
      cr = -0.0161;
    }
    else{
      cr = 0.0145;
    }
      
    v_cmd = vx_tgt * cos(theta_tgt) + vy_tgt * sin(theta_tgt) - cr * w_cmd;

    
    // Clamp to physical limits
    v_cmd = clampf(v_cmd, MPC_MIN_VEL, MPC_MAX_VEL);
    w_cmd = clampf(w_cmd, MPC_MIN_OMEGA, MPC_MAX_OMEGA);
}

//  _buildAndUpdate() — private helper
//
//  Builds Xref (4 x N) by broadcasting the current reference across the horizon,
//  builds Uref (2 x N-1) as zeros (encourage minimal acceleration),
//  then pushes x0, Xref, Uref into the TinyMPC solver.
void TinyMpcController::_buildAndUpdate(const MpcRefState& ref) {
    // Reference world-frame velocity from heading and speed
    const tinytype vx_ref = (tinytype)ref.v * (tinytype)cos((double)ref.theta);
    const tinytype vy_ref = (tinytype)ref.v * (tinytype)sin((double)ref.theta);

    // Build x0 vector (nx x 1)
    tinyVector x0_vec(MPC_NX);
    x0_vec << _x0[0], _x0[1], _x0[2], _x0[3];

    // Build Xref (nx x N): broadcast same reference across all N knotpoints
    tinyMatrix Xref(MPC_NX, MPC_N);
    for (int k = 0; k < MPC_N; ++k) {
        Xref(0, k) = (tinytype)ref.px;
        Xref(1, k) = (tinytype)ref.py;
        Xref(2, k) = vx_ref;
        Xref(3, k) = vy_ref;
    }

    // Build Uref (nu x N-1): zero reference — encourage small accelerations
    tinyMatrix Uref = tinyMatrix::Zero(MPC_NU, MPC_N - 1);

    // Push to solver
    tiny_set_x0(&tiny_solver, x0_vec);
    tiny_set_x_ref(&tiny_solver, Xref);
    tiny_set_u_ref(&tiny_solver, Uref);
}
