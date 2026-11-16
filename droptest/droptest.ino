// LSM6DSOX drop test on the Adafruit ESP32-S3 Reverse TFT Feather, streaming
// CSV for Rerun (see plot_imu.py). IMU on the STEMMA QT port, as in
// smilequest_brush.
#include <Arduino.h>
#include <Wire.h>

#include "lsm6dsox.h"

// ---- I2C bus (Feather STEMMA QT) ------------------------------------------
#define I2C_SDA_GPIO 3
#define I2C_SCL_GPIO 4
#define I2C_POWER_GPIO 7

#define SAMPLE_HZ 100

uint32_t nextSampleUs = 0;

void setup(void) {
  Serial.begin(921600);

  // GPIO7 powers the STEMMA QT connector on the Reverse TFT Feather
  pinMode(I2C_POWER_GPIO, OUTPUT);
  digitalWrite(I2C_POWER_GPIO, HIGH);
  delay(10);
  if (!Wire.begin(I2C_SDA_GPIO, I2C_SCL_GPIO, 400000)) {
    Serial.println("# I2C initialization failed");
  }
  Wire.setTimeOut(50);

  while (!lsm6dsox_init(Wire)) {
    Serial.println("# LSM6DSOX not found; check STEMMA QT wiring");
    delay(1000);
  }

  // Lines starting with '#' are comments; the header names the CSV columns
  Serial.println("# t_us,ax,ay,az,gx,gy,gz,temp");
  nextSampleUs = micros();
}

void loop() {
  if ((int32_t)(micros() - nextSampleUs) < 0) return;
  nextSampleUs += 1000000 / SAMPLE_HZ;

  lsm6dsox_sample s;
  if (!lsm6dsox_read(&s)) {
    Serial.println("# IMU read failed");
    return;
  }

  // accel m/s^2, gyro rad/s, temp C
  Serial.printf("%lu,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%.2f\n",
                (unsigned long)micros(),
                s.ax, s.ay, s.az, s.gx, s.gy, s.gz, s.temp);
}
