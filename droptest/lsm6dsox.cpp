#include "lsm6dsox.h"

#include "i2c_registers.h"
#include <Arduino.h>

// ---- Register map (subset) ------------------------------------------------
#define REG_WHO_AM_I 0x0F
#define REG_CTRL1_XL 0x10 // accelerometer control
#define REG_CTRL2_G 0x11  // gyroscope control
#define REG_CTRL3_C 0x12
#define REG_OUT_TEMP_L 0x20 // temp, then gyro (0x22), then accel (0x28)

#define WHO_AM_I_VALUE 0x6C

// ODR = 104 Hz (0b0100), FS = +/-16g (0b01) -> CTRL1_XL = 0x44.
#define CTRL1_XL_104HZ_16G 0x44
// ODR = 104 Hz (0b0100), FS = 2000 dps (0b11) -> CTRL2_G = 0x4C.
#define CTRL2_G_104HZ_2000DPS 0x4C
// BDU=1 (block data update), IF_INC=1 (auto-increment on multi-byte reads).
#define CTRL3_C_BDU_IFINC 0x44

// Sensitivity for +/-16g full scale: 0.488 mg/LSB.
#define ACCEL_MG_PER_LSB 0.488f
#define MG_TO_MS2 (9.80665f / 1000.0f)
// Sensitivity for 2000 dps full scale: 70 mdps/LSB.
#define GYRO_MDPS_PER_LSB 70.0f
#define MDPS_TO_RADS (PI / 180.0f / 1000.0f)
// Temperature: 256 LSB/degC, 0 LSB = 25 degC.
#define TEMP_LSB_PER_C 256.0f
#define TEMP_OFFSET_C 25.0f

static TwoWire *s_bus = nullptr;

static bool write_reg(uint8_t reg, uint8_t val) {
  return i2c_write_register(s_bus, LSM6DSOX_I2C_ADDR, reg, val);
}

static bool read_regs(uint8_t reg, uint8_t *dst, size_t len) {
  return i2c_read_registers(s_bus, LSM6DSOX_I2C_ADDR, reg, dst, len);
}

static int16_t le16(const uint8_t *p) { return (int16_t)((p[1] << 8) | p[0]); }

bool lsm6dsox_init(TwoWire &bus) {
  s_bus = &bus;

  uint8_t who = 0;
  if (!read_regs(REG_WHO_AM_I, &who, 1) || who != WHO_AM_I_VALUE) {
    return false;
  }

  if (!write_reg(REG_CTRL3_C, CTRL3_C_BDU_IFINC))
    return false;
  if (!write_reg(REG_CTRL1_XL, CTRL1_XL_104HZ_16G))
    return false;
  if (!write_reg(REG_CTRL2_G, CTRL2_G_104HZ_2000DPS))
    return false;
  delay(10);

  return true;
}

bool lsm6dsox_read(lsm6dsox_sample *out) {
  uint8_t raw[14];
  if (!read_regs(REG_OUT_TEMP_L, raw, sizeof(raw))) {
    return false;
  }

  out->temp = le16(&raw[0]) / TEMP_LSB_PER_C + TEMP_OFFSET_C;

  const float g = GYRO_MDPS_PER_LSB * MDPS_TO_RADS;
  out->gx = le16(&raw[2]) * g;
  out->gy = le16(&raw[4]) * g;
  out->gz = le16(&raw[6]) * g;

  const float a = ACCEL_MG_PER_LSB * MG_TO_MS2;
  out->ax = le16(&raw[8]) * a;
  out->ay = le16(&raw[10]) * a;
  out->az = le16(&raw[12]) * a;
  return true;
}
