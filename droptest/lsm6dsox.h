#pragma once

#include <Wire.h>

// LSM6DSOX 6-DoF IMU (Adafruit product 4438), STEMMA QT I2C addr 0x6A.
// Adapted from smilequest_brush; configured wide for drop testing.
#define LSM6DSOX_I2C_ADDR 0x6A

struct lsm6dsox_sample {
  float ax, ay, az; // m/s^2
  float gx, gy, gz; // rad/s
  float temp;       // deg C
};

// Configures accel and gyro for 104 Hz, +/-16g and +/-2000 dps.
//
// Returns true on success, or false if the WHO_AM_I check fails or the
// device cannot be reached on the bus.
bool lsm6dsox_init(TwoWire &bus);

// Reads temperature, gyro and accel in one burst.
bool lsm6dsox_read(lsm6dsox_sample *out);
