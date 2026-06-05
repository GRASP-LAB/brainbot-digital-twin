// circle_tinympc.ino
//
// MCU application: TinyMPC MPC controller over BLE (Nordic UART Service).
//
// Main flow:
// 1. setup(): initialise BLE, MPC controller, and circle trajectory LUT.
// 2. loop(): bridge BLE <-> SerialLP1.
//    - On each received NUS packet (x, y, theta, vx, vy):
//        a. Update robot state in the MPC.
//        b. Get the current reference from the circle trajectory.
//        c. Solve the QP; convert [ax, ay] -> [v_cmd, w_cmd].
//        d. Map velocities to motor PWM integers and send "M<mr>,<ml>\n".
//    - Forward any SerialLP1 bytes back over BLE TX.

#include <Arduino.h>
#include "figure8_trajectory.h"
#include "mpc_controller.h"
#include <STM32duinoBLE.h>

#include "VL_LUT.h"
#include "VR_LUT.h"

#define START_BYTE 0xAA
#define END_BYTE 0x55
#define PACKET_SIZE 13
#define PXM 0.0003788
#define STATUS_LED PC11

HardwareSerial SerialLP1(PC0, PB5);

// Nordic UART Service (NUS) UUIDs
static const char* NUS_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e";
static const char* NUS_RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e";  // Write (PC->STM32)
static const char* NUS_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e";  // Notify (STM32->PC)

uint8_t packetBuffer[PACKET_SIZE];
uint8_t packetIndex = 0;

unsigned long lastPacketTime = 0;
static float g_elapsed_s = 0.0f;


BLEService nusService(NUS_SERVICE_UUID);
BLECharacteristic nusRX(NUS_RX_UUID, BLEWrite | BLEWriteWithoutResponse, 244);
BLECharacteristic nusTX(NUS_TX_UUID, BLENotify, 244);

// TinyMPC controller and circle trajectory
static TinyMpcController mpc;
//static CircleTrajectory g_traj;
static Figure8Trajectory g_traj;

// ---------------------------------------------------------------------------
//  LUT helpers
// ---------------------------------------------------------------------------
static inline int iroundf(float x) {
  return (int)lroundf(x);
}

static uint8_t lookup_VL_nearest(float v, float omega) {
  const float fi = (v - VMIN) / DEL_V;
  const float fj = (omega - OMEGA_MIN) / DEL_OMEGA;

  // Nearest integer index
  int i = iroundf(fi);
  int j = iroundf(fj);

  // Clamp to table bounds
  // i = clampi(i, 0, VL_LUT_NV - 1);
  // j = clampi(j, 0, VL_LUT_NW - 1);
  i = constrain(i, 0, VL_LUT_NV - 1);
  j = constrain(j, 0, VL_LUT_NW - 1);

  // Read LUT
  return VL_LUT[i][j];
}

static uint8_t lookup_VR_nearest(float v, float omega) {
  // Convert to fractional index
  const float fi = (v - VMIN) / DEL_V;
  const float fj = (omega - OMEGA_MIN) / DEL_OMEGA;

  // Nearest integer index
  int i = iroundf(fi);
  int j = iroundf(fj);

  // Clamp to table bounds
  // i = clampi(i, 0, VR_LUT_NV - 1);
  // j = clampi(j, 0, VR_LUT_NW - 1);
  i = constrain(i, 0, VR_LUT_NV - 1);
  j = constrain(j, 0, VR_LUT_NW - 1);

  // Read LUT
  return VR_LUT[i][j];
}



// ---------------------------------------------------------------------------
//  setup()
// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  //while (!Serial) delay(10);
  delay(3000);
  Serial.printf(__FILE__);
  SerialLP1.begin(115200);

  // BLE
  BLE.begin();
  BLE.setLocalName("BLT_NUS");
  BLE.setDeviceName("BLT_NUS");
  nusService.addCharacteristic(nusRX);
  nusService.addCharacteristic(nusTX);
  BLE.addService(nusService);
  BLE.advertise();

  pinMode(STATUS_LED, OUTPUT);
  digitalWrite(STATUS_LED, LOW);

  Serial.println("\n=== TinyMPC Circle Controller ===");

  // Initialise trajectory LUT
  Serial.print("Computing trajectory LUT... ");
  g_traj.init();
  Serial.println("OK");

  // MPC is statically initialised — no extra init() needed
  Serial.println("TinyMPC controller ready.");

  g_elapsed_s = 0.0f;

  Serial.println("Ready!");
  Serial.println();
}

// ---------------------------------------------------------------------------
//  loop()
// ---------------------------------------------------------------------------
void loop() {
  BLEDevice central = BLE.central();

  if (central) {
    SerialLP1.print("Connected: ");
    SerialLP1.println(central.address());

    while (central.connected()) {
      // 1) PC -> STM32 : RX written?
      if (nusRX.written()) {

        unsigned long now = micros();
        unsigned long delta = 0;

        if (lastPacketTime != 0) {
          delta = now - lastPacketTime;
        }

        lastPacketTime = now;

        int n = nusRX.valueLength();
        uint8_t buf[244];

        if (n > 0 && n <= 244) {

          nusRX.readValue(buf, n);

          // Process each byte individually
          for (int i = 0; i < n; i++) {

            uint8_t incoming = buf[i];

            // Wait for start byte
            if (packetIndex == 0) {
              if (incoming != START_BYTE)
                continue;
            }

            packetBuffer[packetIndex++] = incoming;

            // If full packet received
            if (packetIndex == PACKET_SIZE) {
              // Check end byte first
              //Serial.println("Packet size ok");
              if (packetBuffer[PACKET_SIZE - 1] == END_BYTE) {
                int16_t x = packetBuffer[1] | (packetBuffer[2] << 8);
                int16_t y = packetBuffer[3] | (packetBuffer[4] << 8);
                int16_t theta_scaled = packetBuffer[5] | (packetBuffer[6] << 8);
                int16_t vx_scaled = packetBuffer[7] | (packetBuffer[8] << 8);
                int16_t vy_scaled = packetBuffer[9] | (packetBuffer[10] << 8);
                uint8_t checksum = packetBuffer[11];

                uint8_t calc = (x + y + theta_scaled + vx_scaled + vy_scaled) & 0xFF;
                //Serial.println("End ok");
                if (calc == checksum) {

                  //Serial.println("Checksum ok");
                  float theta = theta_scaled / 100.0f;
                  float vx = vx_scaled * PXM;
                  float vy = vy_scaled * PXM;

                  // Serial.print("X: ");
                  // Serial.print(x);
                  // Serial.print(" Y: ");
                  // Serial.print(y);
                  // Serial.print(" T: ");
                  // Serial.print(theta);
                  Serial.print(" Vx: ");
                  Serial.println(vx);
                  Serial.print(" Vy: ");
                  Serial.println(vy);

                  float robot_x = x * PXM;  // Current robot position
                  float robot_y = y * PXM;

                  // ---- Feed state into TinyMPC ----
                  mpc.setState(robot_x, robot_y, vx, vy);

                  // ---- Get current reference from trajectory LUT ----
                  //CircleState ref_k = g_traj.get(g_elapsed_s);
                  Figure8State ref_k = g_traj.get(g_elapsed_s);

                  MpcRefState ref;
                  ref.px = ref_k.px;
                  ref.py = ref_k.py;
                  ref.theta = ref_k.theta;
                  ref.v = ref_k.v;

                  // ---- Solve QP ----
                  float ax = 0.0f, ay = 0.0f;

                  uint32_t start_time = micros();
                  bool ok = mpc.solve(ref, ax, ay);
                  uint32_t inference_time = micros() - start_time;
                  Serial.printf("Inference: %lu us\n", inference_time);
                  Serial.println();

                  if (ok) {
                    // Convert [ax, ay] -> unicycle [v_cmd, w_cmd]
                    float v_cmd = 0.0f, w_cmd = 0.0f;
                    mpc.toUnicycle(ax, ay, theta, v_cmd, w_cmd);

                    Serial.print("Vcmd = ");
                    Serial.println(v_cmd);
                    Serial.print("Wcmd = ");
                    Serial.println(w_cmd);

                    // Map v_cmd, w_cmd -> motor PWM via LUT
                    uint8_t ml = lookup_VL_nearest(v_cmd, w_cmd);
                    uint8_t mr = lookup_VR_nearest(v_cmd, w_cmd);

                    char msg[32] = { 0 };
                    snprintf(msg, sizeof(msg), "M%d,%d\n", ml, mr);
                    Serial.print(msg);
                    SerialLP1.print(msg);

                    // Advance trajectory clock
                    //g_elapsed_s += CIRCLE_DT;
                    g_elapsed_s += F8_DT;

                    // Blink LED to show activity
                    digitalWrite(STATUS_LED, !digitalRead(STATUS_LED));

                  } else {
                    // Solver failed — send safe stop command
                    Serial.print("[TINYMPC] WARN: solver failed, status=");
                    Serial.println(mpc.lastStatus());
                    SerialLP1.print("M0,0\n");
                    digitalWrite(STATUS_LED, LOW);
                  }
                }
              }

              packetIndex = 0;
            }
          }
        }
      }

      // ---- 2) STM32 -> PC : forward SerialLP1 bytes over BLE TX ----
      while (SerialLP1.available()) {
        uint8_t out[244];
        int n = SerialLP1.readBytes(out, sizeof(out));
        if (n > 0) nusTX.writeValue(out, n);
      }

      delay(1);
    }

    SerialLP1.println("Disconnected");
  }
}
