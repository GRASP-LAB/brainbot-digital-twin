#include <math.h>
#include <stdint.h>
#include "VL_LUT.h"
#include "VR_LUT.h"
#include <stdio.h>

// Clamp helper if constrain() doesnt work
static inline int clampi(int x, int lo, int hi) {
  return (x < lo) ? lo : (x > hi) ? hi : x;
}
static inline int iroundf(float x) {
  return (int)lroundf(x);
}

uint8_t lookup_VL_nearest(float v, float omega) {
  // Convert to fractional index
  const float fi = (v     - VMIN) / DEL_V;
  const float fj = (omega - OMEGA_MIN) / DEL_OMEGA;

  // Nearest integer index
  int i = iroundf(fi);
  int j = iroundf(fj);

  // Clamp to table bounds
  i = clampi(i, 0, VL_LUT_NV - 1);
  j = clampi(j, 0, VL_LUT_NW - 1);
  // i = constrain(i, 0, VL_LUT_NV - 1);
  // j = constrain(j, 0, VL_LUT_NW - 1);

  // Read LUT
  return VL_LUT[i][j];
}

uint8_t lookup_VR_nearest(float v, float omega) 
{
  // Convert to fractional index
  const float fi = (v     - VMIN) / DEL_V;
  const float fj = (omega - OMEGA_MIN) / DEL_OMEGA;

  // Nearest integer index
  int i = iroundf(fi);
  int j = iroundf(fj);

  // Clamp to table bounds
  i = clampi(i, 0, VR_LUT_NV - 1);
  j = clampi(j, 0, VR_LUT_NW - 1);
  // i = constrain(i, 0, VR_LUT_NV - 1);
  // j = constrain(j, 0, VR_LUT_NW - 1);

  // Read LUT
  return VR_LUT[i][j];
}

void setup() {
  // Serial.begin(115200);

  float v_q = 0.1f;
  float w_q = -1.0f;

  uint8_t VL = lookup_VL_nearest(v_q, w_q);
  uint8_t VR = lookup_VR_nearest(v_q, w_q);
  // printf("VL nearest = ");
  printf("VL = %d\n", VL);
  printf("VR nearest = %d\n", VR);
  // println(VR);
}

void loop() {}

void main()
{

  setup();
  // while (1) {
  //   loop();
  // }
}