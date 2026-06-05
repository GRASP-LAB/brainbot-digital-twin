/*
 * mpc_controller.h
 * 
 * Summary:
 * This file defines the TinyMpcController class. It acts as the brain for the robot,
 * helping it move along a circular path. It takes the robot's current position and
 * speed (state), looks at where the robot needs to go (horizon), and calculates the 
 * best acceleration commands to get there safely while following speed limits.
 * 
 * Flow:
 * 1. Initialize the controller trajectory LUT (init).
 * 2. Constantly feed it the robot's current state (setState).
 * 3. Ask it to solve for the next best moves based on time (solve).
 * 4. Convert its acceleration answers into direct motor speeds (toUnicycle).
 *
 * State  : [px, py, vx, vy]
 * Input  : [ax, ay]  (world-frame accelerations)
 * Horizon: N = 10 , dt = 0.1 s
 *
 * Generated workspace dimensions (tiny_data.cpp):
 *   Xref = 4 x 10   (nx x N)
 *   Uref = 2 x 9    (nu x N-1)
 *   solution->u = 2 x 9
 *
 * Cost weights (from tinympc_code_generator_for_mcu.py):
 *   Q  = diag(10, 10, 1, 1)   (+ rho -> stored as 11, 11, 2, 2 in workspace)
 *   R  = 0.1 * I2             (+ rho -> stored as 1.1, 1.1 in workspace)
 *
 * Constraints:
 *   |vx|, |vy| <= 0.15 m/s
 *   |ax|, |ay| <= 2.37 m/s^2
 */

#ifndef MPC_CONTROLLER_H_
#define MPC_CONTROLLER_H_

#include <stdint.h>
#include <math.h>

// TinyMPC generated headers
#include "tiny_api.hpp"   // tiny_solve, tiny_set_x0/x_ref/u_ref
#include "tiny_data.hpp"  // extern TinySolver tiny_solver;

// Generic reference state (trajectory-agnostic)
struct MpcRefState {
    float px;
    float py;
    float theta;
    float v;
};

static constexpr float MPC_DT = 0.1f;  // Ts used in MPC

// MPC problem dimensions (must match tiny_data.cpp)
static constexpr int MPC_NX     = 4;   // states: [px, py, vx, vy]
static constexpr int MPC_NU     = 2;   // inputs: [ax, ay]
static constexpr int MPC_N      = 10;  // knotpoints (horizon length)

// Constraint bounds (match generator script)
//static constexpr float MPC_MIN_VEL   = 0.048f;   // m/s
//static constexpr float MPC_MAX_VEL   = 0.115f;   // m/s
static constexpr float MPC_MIN_VEL   = 0.034f;   // m/s
static constexpr float MPC_MAX_VEL   = 0.135f;   // m/s
static constexpr float MPC_MAX_ACCEL = 1.0f;   // m/s^2
static constexpr float MPC_MAX_OMEGA = 1.43f;    // rad/s (unicycle angular rate limit)
static constexpr float MPC_MIN_OMEGA = -3.46f;    // rad/s (unicycle angular rate limit)

class TinyMpcController {
public:
    TinyMpcController();

    // Call once in setup() to pre-compute the circle LUT
    void init();

    // Feed current robot state; call every tick before solve()
    // x0 = [px, py, vx, vy]  (world frame)
    void setState(float px, float py, float vx, float vy);

    // Run one MPC solve for the given elapsed time.
    // Returns true if solver succeeded and output is valid.
    // ax_out, ay_out: optimal world-frame accelerations (first control step)
    bool solve(const MpcRefState& ref, float& ax_out, float& ay_out);

    // Convert world-frame [ax, ay] + current state to unicycle commands.
    // v_cmd:  body-frame linear  velocity command [m/s]
    // w_cmd:  body-frame angular velocity command [rad/s]
    void toUnicycle(float ax, float ay, float yaw,
                    float& v_cmd, float& w_cmd) const;

    // Last solve status (0 = solved)
    int lastStatus() const { return _last_status; }

    // reference is passed in via solve()

private:
    // Build Xref and Uref from current reference, push to TinyMPC
    void _buildAndUpdate(const MpcRefState& ref);

    // Current robot state [px, py, vx, vy] as double (tinytype)
    tinytype _x0[MPC_NX];

    // Last solve return code
    int _last_status;
};

#endif  // MPC_CONTROLLER_H_
