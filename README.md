# ysafetyml

Fall detection from waist-worn IMU data, using the [SisFall](https://doi.org/10.3390/s17010198) dataset, plus LSM6DSOX drop-test firmware.

## Layout

| Path | What |
|---|---|
| `data/sisfall_parquet/` | The full SisFall dataset (4505 trials, 38 subjects), one parquet file per subject with the raw int16 sensor counts |
| `src/ysafetyml/sisfall.py` | `load_sisfall()`: every sample as one DataFrame in g and deg/s, with activity / subject / trial labels; `subjects()`: subject demographics |
| `src/ysafetyml/motion.py` | Reconstructs waist orientation and position from one trial (orientation filter, gravity removal, double integration with zero-velocity updates) |
| `notebooks/sisfall2.ipynb` | Windowing + features, HistGradientBoosting vs. a GRU, evaluated on held-out subjects |
| `scripts/rerun_sisfall.py` | Replays every subject in [Rerun](https://rerun.io) as a sphere moving through the room |
| `droptest/` | Arduino/ESP32 firmware and a Rerun live plotter for an LSM6DSOX drop test |
| `particle/` | Particle firmware project |

## Setup

```bash
uv sync
uv run python -c "from ysafetyml.sisfall import load_sisfall; print(load_sisfall().shape)"
uv run scripts/rerun_sisfall.py --subjects SA01
```

To rebuild the parquet from the original SisFall text files, put them in `data/sisfall/<SUBJECT>/` and call `ysafetyml.sisfall.write_parquet()`.

## Data

SisFall: A. Sucerquia, J. D. López, J. F. Vargas-Bonilla, "SisFall: A Fall and Movement Dataset", *Sensors* 17(1):198, 2017. Data are from 3 sensors (ADXL345 and MMA8451Q accelerometers, ITG3200 gyroscope) sampled at 200 Hz. The subject folder `SA15/` in the original release contains SA15's D17 trials misnamed as `D17_SE15_R0x.txt`; the loader assigns them to SA15.
