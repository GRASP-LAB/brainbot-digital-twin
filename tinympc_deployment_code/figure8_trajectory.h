#ifndef FIGURE8_TRAJECTORY_H_
#define FIGURE8_TRAJECTORY_H_

// Equations:
// x(t) = A * sin(w*t)
// y(t) = 0.5 * B * sin(2*w*t) * (1 - alpha * sin^2 (w*t))
// where w = 2*pi / period

#include <stdint.h>
#include <math.h>

// Arena & control settings
static constexpr float F8_A        = 0.15f;   // X amplitude
static constexpr float F8_B        = 0.30f;   // Y amplitude
static constexpr float F8_PERIOD   = 28.0f;   // Seconds per loop
static constexpr float F8_DT       = 0.1f;    // MPC Ts
static constexpr float F8_CENTER_X = 0.0f;
static constexpr float F8_CENTER_Y = 0.0f;
static constexpr float F8_alpha    = 0.0f;    // To make the fig 8 rounder

// LUT size
static constexpr int F8_STEPS = static_cast<int>(F8_PERIOD / F8_DT);

// Reference state format
struct Figure8State {
    float px;
    float py;
    float theta;
    float v;
    float om;
};

// Generates the figure-8 path
class Figure8Trajectory {
public:
    // Pre-compute the path points (call once in setup)
    void init() {
        const float omega = 2.0f * M_PI / F8_PERIOD;

        for (int i = 0; i < F8_STEPS; ++i) {
            const float t     = i * F8_DT;
            const float wt    = omega * t;
            const float w2t   = 2.0f * wt;

            const float swt = sinf(wt); 
            const float cwt = cosf(wt);
            const float sw2t = sinf(w2t);
            const float cw2t = cosf(w2t);

            // Position
            _lut[i].px = F8_A * swt  + F8_CENTER_X;
            _lut[i].py = 0.5f * F8_B * sw2t + F8_CENTER_Y; //* (1 - F8_alpha * ( swt * swt ) )  + F8_CENTER_Y;

            // First derivative (Velocity)
            const float vx = F8_A * omega * cwt;
            const float vy = F8_B * omega * cw2t; //*(1 - F8_alpha * swt * swt) - F8_alpha * swt * sw2t *cw2t );

            // Second derivative (Acceleration)
            const float ax = -F8_A * omega * omega * swt;
            const float ay = -F8_B * omega * omega * 2 * sw2t ; //* ( 4 * F8_alpha * cwt * cw2t * swt + F8_alpha * cwt * cwt * sw2t + (2 - 3 * F8_alpha * swt * swt ) *sw2t );

            // Heading & Linear speed
            _lut[i].theta = atan2f(vy, vx);
            _lut[i].v = sqrtf(vx * vx + vy * vy);

            // Angular speed (using curvature formula)
            const float v2 = vx * vx + vy * vy;
            _lut[i].om = (v2 > 1e-6f) ? (vx * ay - vy * ax) / v2 : 0.0f;
        }
    }

    // Get reference state at a given time (auto-loops)
    Figure8State get(float time_s) const {
        float t_mod = fmodf(time_s, F8_PERIOD);
        if (t_mod < 0.0f) t_mod += F8_PERIOD;

        int idx = static_cast<int>(t_mod / F8_DT);
        if (idx >= F8_STEPS) idx = F8_STEPS - 1;

        return _lut[idx];
    }

    // Get LUT index for a time
    int indexAt(float time_s) const {
        float t_mod = fmodf(time_s, F8_PERIOD);
        if (t_mod < 0.0f) t_mod += F8_PERIOD;
        int idx = static_cast<int>(t_mod / F8_DT);
        if (idx >= F8_STEPS) idx = F8_STEPS - 1;
        return idx;
    }

    // Direct access to LUT
    const Figure8State& atIndex(int idx) const {
        idx = idx % F8_STEPS;
        if (idx < 0) idx += F8_STEPS;
        return _lut[idx];
    }

private:
    Figure8State _lut[F8_STEPS];
};

#endif  // FIGURE8_TRAJECTORY_H_
