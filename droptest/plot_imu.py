# /// script
# requires-python = ">=3.10"
# dependencies = ["rerun-sdk==0.36.0", "pyserial>=3.5"]
# ///
"""Stream LSM6DSOX CSV from the ESP32 serial port into Rerun.

Usage: uv run plot_imu.py [/dev/ttyACM0] [--baud 921600]
"""
import argparse
import math

import rerun as rr
import serial

SERIES = {
    "imu/accel": (["ax", "ay", "az"], [[230, 80, 80], [80, 200, 80], [80, 120, 230]]),
    "imu/gyro": (["gx", "gy", "gz"], [[230, 80, 80], [80, 200, 80], [80, 120, 230]]),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("port", nargs="?", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=921600)
    args = parser.parse_args()

    rr.init("droptest", spawn=True)
    for path, (names, colors) in SERIES.items():
        rr.log(path, rr.SeriesLines(names=names, colors=colors), static=True)
    rr.log("imu/accel_mag", rr.SeriesLines(names="|a|"), static=True)
    rr.log("imu/temp", rr.SeriesLines(names="temp"), static=True)

    last_us = None
    offset_us = 0
    with serial.Serial(args.port, args.baud, timeout=1) as ser:
        ser.reset_input_buffer()
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue
            if line.startswith("#"):
                print(line)
                continue
            try:
                t_us, *v = (float(x) for x in line.split(","))
            except ValueError:
                continue
            if len(v) != 7:
                continue

            # micros() wraps every ~71 minutes
            if last_us is not None and t_us < last_us:
                offset_us += 2**32
            last_us = t_us
            rr.set_time("device_time", duration=(t_us + offset_us) * 1e-6)

            ax, ay, az, gx, gy, gz, temp = v
            rr.log("imu/accel", rr.Scalars([ax, ay, az]))
            rr.log("imu/gyro", rr.Scalars([gx, gy, gz]))
            # ~9.81 at rest, ~0 in free fall, spikes on impact
            rr.log("imu/accel_mag", rr.Scalars(math.sqrt(ax * ax + ay * ay + az * az)))
            rr.log("imu/temp", rr.Scalars(temp))


if __name__ == "__main__":
    main()
