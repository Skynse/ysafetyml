#pragma once

#include <Wire.h>

// Transactions run only on the Arduino loop task, never inside BLE callbacks.
inline bool i2c_write_register(TwoWire *bus, uint8_t address, uint8_t reg,
                               uint8_t value) {
  if (bus == nullptr)
    return false;
  bus->beginTransmission(address);
  bus->write(reg);
  bus->write(value);
  return bus->endTransmission() == 0;
}

inline bool i2c_read_registers(TwoWire *bus, uint8_t address, uint8_t reg,
                               uint8_t *output, size_t size) {
  if (bus == nullptr || output == nullptr)
    return false;
  bus->beginTransmission(address);
  bus->write(reg);
  if (bus->endTransmission(false) != 0)
    return false;
  if (bus->requestFrom(address, size) != size)
    return false;
  for (size_t i = 0; i < size; ++i)
    output[i] = bus->read();
  return true;
}
