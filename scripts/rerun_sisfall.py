"""Replay every SisFall subject in Rerun as a sphere moving through the room.

The sphere is the waist-worn IMU. Its path is reconstructed from the
accelerometer and gyro (see ysafetyml.motion: orientation filter, gravity
removal, double integration with zero-velocity updates), so it is an estimate:
vertical movement such as sitting, squatting and falling shows up well, while
walking and jogging bob roughly in place, because forward progress can't be
recovered without a magnetometer or position reference.

Each subject gets its own recording (pick it in the viewer's recording list),
with their trials played back to back on the ``time`` timeline. Every trial
starts at the origin, at an approximate waist height for the activity's starting
posture (standing / seated / lying).

- sphere + body axes (x red, y green, z blue): IMU position and orientation
- trail: the last 2 s of the path, colored by |a| (blue free fall -> white 1 g -> red impact)
- plots: waist height, |a|, raw accel and gyro, and a log of the activity playing

Usage:
    uv run scripts/rerun_sisfall.py                       # all subjects, spawn viewer
    uv run scripts/rerun_sisfall.py --subjects SA01 SE06  # a few subjects
    uv run scripts/rerun_sisfall.py --activities F01 F05 D07
    uv run scripts/rerun_sisfall.py --save recordings/    # one .rrd per subject
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from ysafetyml.motion import reconstruct
from ysafetyml.sisfall import ACTIVITIES, SAMPLE_HZ, load_sisfall, subjects

APP_ID = "sisfall"
GAP_S = 1.0  # pause between consecutive trials on the timeline
TRAIL_S = 2.0
PLOT_WINDOW_S = (-8.0, 4.0)  # plots scroll with the playhead instead of showing the whole session
SPHERE_RADIUS = 0.08  # m
XYZ_COLORS = [[230, 80, 80], [80, 200, 80], [80, 120, 230]]

# Rough waist height as a fraction of body height, by starting posture.
STANDING, SEATED, LYING = 0.53, 0.30, 0.12
START_POSTURE = {"D11": SEATED, "D12": SEATED, "D13": SEATED, "D14": LYING,
                 "F08": SEATED, "F09": SEATED, "F13": SEATED, "F14": SEATED, "F15": SEATED}  # fmt: skip

# |a| -> color: blue in free fall, white at 1 g, red at 4 g and above
_ACC_CMAP = mpl.colormaps["coolwarm"]
_ACC_NORM = mpl.colors.TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=4.0)


def floor_grid(half: float = 2.0, step: float = 0.5) -> list[np.ndarray]:
    ticks = np.arange(-half, half + 1e-9, step)
    return [np.array([[t, -half, 0], [t, half, 0]]) for t in ticks] + [
        np.array([[-half, t, 0], [half, t, 0]]) for t in ticks
    ]


def blueprint() -> rrb.Blueprint:
    trail = rrb.VisibleTimeRange(
        "time",
        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-TRAIL_S),
        end=rrb.TimeRangeBoundary.cursor_relative(),
    )
    window = rrb.VisibleTimeRange(
        "time",
        start=rrb.TimeRangeBoundary.cursor_relative(seconds=PLOT_WINDOW_S[0]),
        end=rrb.TimeRangeBoundary.cursor_relative(seconds=PLOT_WINDOW_S[1]),
    )
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="world",
                name="Waist movement",
                time_ranges=trail,
                eye_controls=rrb.EyeControls3D(position=[2.5, -2.5, 1.6], look_target=[0, 0, 0.5], eye_up=[0, 0, 1]),
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(origin="plots/height", name="Waist height [m]", time_ranges=window),
                rrb.TimeSeriesView(origin="plots/accel_mag", name="|a| [g]", time_ranges=window),
                rrb.TimeSeriesView(origin="plots/accel", name="Accel MMA8451Q [g]", time_ranges=window),
                rrb.TimeSeriesView(origin="plots/gyro", name="Gyro ITG3200 [deg/s]", time_ranges=window),
                rrb.TextLogView(origin="activity", name="Activity"),
            ),
            column_shares=[3, 2],
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        # Paused, not the default "following": while data streams in, following jumps the
        # playhead to the newest sample, racing through hours of trials in seconds.
        # Press play (space) to watch at real time.
        rrb.TimePanel(state="expanded", timeline="time", playback_speed=1.0, play_state="paused"),
    )


def log_static(rec: rr.RecordingStream) -> None:
    rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rec.log("world/floor", rr.LineStrips3D(floor_grid(), colors=[90, 90, 90], radii=0.003), static=True)
    rec.log(
        "world/imu/sphere",
        rr.Ellipsoids3D(radii=[SPHERE_RADIUS], colors=[240, 180, 60], fill_mode=rr.components.FillMode.Solid),
        static=True,
    )
    rec.log("world/imu/axes", rr.Arrows3D(vectors=np.eye(3) * 0.25, colors=XYZ_COLORS, radii=0.008), static=True)
    rec.log("plots/height", rr.SeriesLines(names="z", colors=[60, 180, 200]), static=True)
    rec.log("plots/accel_mag", rr.SeriesLines(names="|a|", colors=[220, 160, 40]), static=True)
    rec.log("plots/accel", rr.SeriesLines(names=["ax", "ay", "az"], colors=XYZ_COLORS), static=True)
    rec.log("plots/gyro", rr.SeriesLines(names=["gx", "gy", "gz"], colors=XYZ_COLORS), static=True)


def log_subject(rec: rr.RecordingStream, subject_df, height_m: float, step: int) -> None:
    """Reconstruct each trial at the full 200 Hz, then send every ``step``-th sample as columns."""
    times, pos, quat, acc, gyr = [], [], [], [], []
    trial_starts, trial_texts, trial_levels = [], [], []
    offset = 0.0
    for trial_id, trial in subject_df.groupby("trial_id", observed=True, sort=True):
        activity = trial["activity"].iloc[0]
        a = trial[["ax_mma", "ay_mma", "az_mma"]].to_numpy()
        g = trial[["gx", "gy", "gz"]].to_numpy()
        p, q, _ = reconstruct(a, g)
        p[:, 2] += START_POSTURE.get(activity, STANDING) * height_m

        t = trial["t"].to_numpy(np.float64)[::step] + offset
        times.append(t)
        pos.append(p[::step])
        quat.append(q[::step])
        acc.append(a[::step])
        gyr.append(g[::step])
        trial_starts.append(offset)
        trial_texts.append(f"{trial_id}: {ACTIVITIES[activity]}")
        trial_levels.append(rr.TextLogLevel.WARN if activity.startswith("F") else rr.TextLogLevel.INFO)
        offset = t[-1] + GAP_S

    t = np.concatenate(times)
    pos = np.concatenate(pos)
    quat = np.concatenate(quat)
    acc = np.concatenate(acc)
    gyr = np.concatenate(gyr)
    mag = np.linalg.norm(acc, axis=1)
    colors = (_ACC_CMAP(_ACC_NORM(mag))[:, :3] * 255).astype(np.uint8)
    time = rr.TimeColumn("time", duration=t)

    rec.send_columns("world/imu", indexes=[time], columns=rr.Transform3D.columns(translation=pos, quaternion=quat))
    rec.send_columns(
        "world/trail",
        indexes=[time],
        columns=rr.Points3D.columns(positions=pos, colors=colors, radii=np.full(len(t), 0.015)),
    )
    rec.send_columns("plots/height", indexes=[time], columns=rr.Scalars.columns(scalars=pos[:, 2]))
    rec.send_columns("plots/accel_mag", indexes=[time], columns=rr.Scalars.columns(scalars=mag))
    rec.send_columns("plots/accel", indexes=[time], columns=rr.Scalars.columns(scalars=acc).partition(np.full(len(t), 3)))
    rec.send_columns("plots/gyro", indexes=[time], columns=rr.Scalars.columns(scalars=gyr).partition(np.full(len(t), 3)))
    rec.send_columns(
        "activity",
        indexes=[rr.TimeColumn("time", duration=trial_starts)],
        columns=rr.TextLog.columns(text=trial_texts, level=trial_levels),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subjects", nargs="+", help="subject ids, e.g. SA01 SE06 (default: all)")
    parser.add_argument("--activities", nargs="+", help="activity codes, e.g. F01 D07 (default: all)")
    parser.add_argument("--hz", type=int, default=50, help=f"playback sample rate, divides {SAMPLE_HZ} (default: 50)")
    parser.add_argument("--save", type=Path, help="write <dir>/<subject>.rrd instead of spawning a viewer")
    args = parser.parse_args()

    if SAMPLE_HZ % args.hz:
        parser.error(f"--hz must divide {SAMPLE_HZ}")
    step = SAMPLE_HZ // args.hz

    df = load_sisfall()
    if args.activities:
        df = df[df["activity"].isin(args.activities)]
    heights = subjects()["height_cm"] / 100
    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)

    for subject in args.subjects or sorted(df["subject"].unique()):
        subject_df = df[df["subject"] == subject]
        if subject_df.empty:
            print(f"{subject}: no trials, skipping")
            continue
        rec = rr.RecordingStream(APP_ID, recording_id=subject)
        if args.save:
            rec.save(args.save / f"{subject}.rrd")
        else:
            rec.spawn()  # reuses the viewer if one is already running
        # Sent as the active blueprint before any data, so it overrides a layout the viewer
        # remembers from an earlier run and the playhead stays at the start.
        rec.send_blueprint(blueprint(), make_active=True)
        rec.send_recording_name(subject)
        log_static(rec)
        log_subject(rec, subject_df, heights[subject], step)
        rec.flush()
        print(f"{subject}: {subject_df['trial_id'].nunique()} trials")


if __name__ == "__main__":
    main()
