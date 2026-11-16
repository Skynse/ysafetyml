"""Loader for the full raw SisFall dataset (data/sisfall/<SUBJECT>/<CODE>_<SUBJECT>_<TRIAL>.txt).

Each file holds one trial: 9 comma-separated integer columns per row, sampled at
200 Hz, with a trailing ``;`` on every line (see data/sisfall/Readme.txt):

- cols 1-3: ADXL345 accel x/y/z   (13 bit, +-16 g)
- cols 4-6: ITG3200 gyro x/y/z    (16 bit, +-2000 deg/s)
- cols 7-9: MMA8451Q accel x/y/z  (14 bit, +-8 g)

``load_sisfall`` returns one long DataFrame (one row per sample) with the raw
counts converted to g and deg/s, plus trial labels and subject metadata.

The raw text files aren't needed once parsed: ``data/sisfall_parquet/`` holds the
same data (one file per subject, raw int16 counts, ~175 MB in total) and is what
``load_sisfall`` reads. It is committed to the repo; the raw files are not.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SISFALL_DIR = PROJECT_ROOT / "data" / "sisfall"
PARQUET_DIR = PROJECT_ROOT / "data" / "sisfall_parquet"

SAMPLE_HZ = 200

# ADXL345 first so the first six match ysafetyml.data.CHANNELS.
CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz", "ax_mma", "ay_mma", "az_mma")

# Readme: value = (2 * range / 2**resolution) * counts
_ADXL345 = 2 * 16 / 2**13  # g per count
_ITG3200 = 2 * 2000 / 2**16  # deg/s per count
_MMA8451Q = 2 * 8 / 2**14  # g per count
SCALE = np.array([_ADXL345] * 3 + [_ITG3200] * 3 + [_MMA8451Q] * 3, dtype=np.float32)

ACTIVITIES = {
    "D01": "Walking slowly",
    "D02": "Walking quickly",
    "D03": "Jogging slowly",
    "D04": "Jogging quickly",
    "D05": "Walking upstairs and downstairs slowly",
    "D06": "Walking upstairs and downstairs quickly",
    "D07": "Slowly sit in a half height chair, wait a moment, and up slowly",
    "D08": "Quickly sit in a half height chair, wait a moment, and up quickly",
    "D09": "Slowly sit in a low height chair, wait a moment, and up slowly",
    "D10": "Quickly sit in a low height chair, wait a moment, and up quickly",
    "D11": "Sitting a moment, trying to get up, and collapse into a chair",
    "D12": "Sitting a moment, lying slowly, wait a moment, and sit again",
    "D13": "Sitting a moment, lying quickly, wait a moment, and sit again",
    "D14": "Being on one's back change to lateral position, wait a moment, and change to one's back",
    "D15": "Standing, slowly bending at knees, and getting up",
    "D16": "Standing, slowly bending without bending knees, and getting up",
    "D17": "Standing, get into a car, remain seated and get out of the car",
    "D18": "Stumble while walking",
    "D19": "Gently jump without falling (trying to reach a high object)",
    "F01": "Fall forward while walking caused by a slip",
    "F02": "Fall backward while walking caused by a slip",
    "F03": "Lateral fall while walking caused by a slip",
    "F04": "Fall forward while walking caused by a trip",
    "F05": "Fall forward while jogging caused by a trip",
    "F06": "Vertical fall while walking caused by fainting",
    "F07": "Fall while walking, with use of hands in a table to dampen fall, caused by fainting",
    "F08": "Fall forward when trying to get up",
    "F09": "Lateral fall when trying to get up",
    "F10": "Fall forward when trying to sit down",
    "F11": "Fall backward when trying to sit down",
    "F12": "Lateral fall when trying to sit down",
    "F13": "Fall forward while sitting, caused by fainting or falling asleep",
    "F14": "Fall backward while sitting, caused by fainting or falling asleep",
    "F15": "Lateral fall while sitting, caused by fainting or falling asleep",
}

# subject: (age, height_cm, weight_kg, gender)
_SUBJECTS = {
    "SA01": (26, 165, 53, "F"), "SA02": (23, 176, 58.5, "M"), "SA03": (19, 156, 48, "F"),
    "SA04": (23, 170, 72, "M"), "SA05": (22, 172, 69.5, "M"), "SA06": (21, 169, 58, "M"),
    "SA07": (21, 156, 63, "F"), "SA08": (21, 149, 41.5, "F"), "SA09": (24, 165, 64, "M"),
    "SA10": (21, 177, 67, "M"), "SA11": (19, 170, 80.5, "M"), "SA12": (25, 153, 47, "F"),
    "SA13": (22, 157, 55, "F"), "SA14": (27, 160, 46, "F"), "SA15": (25, 160, 52, "F"),
    "SA16": (20, 169, 61, "F"), "SA17": (23, 182, 75, "M"), "SA18": (23, 181, 73, "M"),
    "SA19": (30, 170, 76, "M"), "SA20": (30, 150, 42, "F"), "SA21": (30, 183, 68, "M"),
    "SA22": (19, 158, 50.5, "F"), "SA23": (24, 156, 48, "F"),
    "SE01": (71, 171, 102, "M"), "SE02": (75, 150, 57, "F"), "SE03": (62, 150, 51, "F"),
    "SE04": (63, 160, 59, "F"), "SE05": (63, 165, 72, "M"), "SE06": (60, 163, 79, "M"),
    "SE07": (65, 168, 76, "M"), "SE08": (68, 163, 72, "F"), "SE09": (66, 167, 65, "M"),
    "SE10": (64, 156, 66, "F"), "SE11": (66, 169, 63, "F"), "SE12": (69, 164, 56.5, "M"),
    "SE13": (65, 171, 72.5, "M"), "SE14": (67, 163, 58, "M"), "SE15": (64, 150, 50, "F"),
}  # fmt: skip

_FILENAME = re.compile(r"^([DF]\d{2})_(S[AE]\d{2})_R(\d{2})\.txt$")


def subjects() -> pd.DataFrame:
    """Subject metadata from the Readme, indexed by subject id."""
    df = pd.DataFrame.from_dict(
        _SUBJECTS, orient="index", columns=["age", "height_cm", "weight_kg", "gender"]
    )
    df.index.name = "subject"
    df["elderly"] = df.index.str.startswith("SE")
    return df


def read_counts(path: Path) -> np.ndarray:
    """One trial file as (n, 9) int16 raw sensor counts, columns in ``CHANNELS`` order."""
    # Lines end in ";\r\n"; treating ";" as a comment drops it so the last column parses as int.
    return pd.read_csv(
        path, header=None, names=list(CHANNELS), dtype=np.int16, skipinitialspace=True, comment=";"
    ).to_numpy()


def read_trial(path: Path) -> np.ndarray:
    """One trial file as (n, 9) float32 in g / deg/s, columns in ``CHANNELS`` order."""
    return read_counts(path).astype(np.float32) * SCALE


def trial_files(root: Path = SISFALL_DIR) -> pd.DataFrame:
    """Index of trial files: one row per file with activity, subject, trial and path."""
    rows = []
    for path in sorted(root.glob("S[AE]*/*.txt")):
        m = _FILENAME.match(path.name)
        if m is None:
            raise ValueError(f"unexpected file name: {path}")
        activity, _, trial = m.groups()
        # Subject comes from the folder: SA15/ ships its D17 trials misnamed as
        # D17_SE15_R0x.txt (contents differ from the real SE15/ files).
        rows.append((activity, path.parent.name, int(trial), path))
    return pd.DataFrame(rows, columns=["activity", "subject", "trial", "path"])


def write_parquet(root: Path = SISFALL_DIR, out: Path = PARQUET_DIR) -> None:
    """Parse the raw text files into ``out/<SUBJECT>.parquet`` (activity, trial, raw counts)."""
    files = trial_files(root)
    out.mkdir(parents=True, exist_ok=True)
    for subject, group in files.groupby("subject", sort=True):
        arrays = [read_counts(p) for p in group["path"]]
        lengths = [len(a) for a in arrays]
        df = pd.DataFrame(np.concatenate(arrays), columns=list(CHANNELS))
        df.insert(0, "activity", np.repeat(group["activity"].to_numpy(), lengths))
        df.insert(1, "trial", np.repeat(group["trial"].to_numpy(np.int8), lengths))
        df.to_parquet(out / f"{subject}.parquet", index=False, compression="zstd", compression_level=19)


def load_sisfall(parquet_dir: Path = PARQUET_DIR, root: Path = SISFALL_DIR) -> pd.DataFrame:
    """All trials as one long DataFrame, one row per 200 Hz sample.

    Columns: ``trial_id`` (e.g. "F05_SA01_R04"), ``activity``, ``subject``, ``trial``,
    ``sample`` (index within trial), ``t`` (seconds), ``is_fall``, then ``CHANNELS``
    (accel in g, gyro in deg/s). Join ``subjects()`` on ``subject`` for demographics.

    Reads ``parquet_dir``; if it is missing, builds it from the raw files in ``root`` first.
    """
    paths = sorted(parquet_dir.glob("S[AE]*.parquet"))
    if not paths:
        if not root.is_dir():
            raise FileNotFoundError(f"neither {parquet_dir} nor the raw dataset in {root} exists")
        write_parquet(root, parquet_dir)
        paths = sorted(parquet_dir.glob("S[AE]*.parquet"))

    parts = [pd.read_parquet(p).assign(subject=p.stem) for p in paths]
    raw = pd.concat(parts, ignore_index=True)
    del parts

    meta = raw[["activity", "subject", "trial"]].copy()
    meta.insert(0, "trial_id", meta["activity"] + "_" + meta["subject"] + "_R" + meta["trial"].map("{:02d}".format))
    # rows of a trial are contiguous, so the sample index restarts wherever trial_id changes
    new_trial = np.r_[True, meta["trial_id"].to_numpy()[1:] != meta["trial_id"].to_numpy()[:-1]]
    starts = np.flatnonzero(new_trial)
    meta["sample"] = (np.arange(len(meta)) - np.repeat(starts, np.diff(np.r_[starts, len(meta)]))).astype(np.int32)
    meta["t"] = (meta["sample"] / SAMPLE_HZ).astype(np.float32)
    meta["is_fall"] = meta["activity"].str.startswith("F")
    for col in ("trial_id", "activity", "subject"):
        meta[col] = meta[col].astype("category")
    meta["trial"] = meta["trial"].astype(np.int8)

    values = raw[list(CHANNELS)].to_numpy(np.float32) * SCALE
    return pd.concat([meta, pd.DataFrame(values, columns=list(CHANNELS))], axis=1)
