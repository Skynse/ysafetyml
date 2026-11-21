"""Animated walkthrough of the SisFall fall-detection pipeline, built with Manim.

Uses real data (subject SA01) for the signal, window and motion scenes; the model
numbers are the held-out results from notebooks/sisfall2.ipynb.

Manim isn't a project dependency, so pull it in for the run (no LaTeX needed):
    uv run --with manim manim -qm scripts/manim_sisfall.py SisFallPipeline   # 720p preview
    uv run --with manim manim -qh scripts/manim_sisfall.py SisFallPipeline   # 1080p
"""

import numpy as np
import pandas as pd
from manim import *

from ysafetyml.motion import reconstruct
from ysafetyml.sisfall import CHANNELS, PARQUET_DIR, SAMPLE_HZ, SCALE

config.background_color = "#101418"
Text.set_default(font="Inter")

ADL_C = "#4FC3A1"
FALL_C = "#FF5C5C"
ACC_C = "#F2B544"
HGB_C = "#5DA9F6"
GRU_C = "#C98BF0"
DIM = "#6B7580"

WINDOW, STRIDE = 512, 256
STANDING = 0.53 * 1.65  # waist height (m) of SA01, as in scripts/rerun_sisfall.py

# Held-out results from notebooks/sisfall2.ipynb (8 test subjects, 8,565 windows)
RESULTS = {
    "HGB": {"acc": 0.9966, "bal": 0.9951, "recall": 0.9933, "precision": 0.9688, "tn": 7792, "fp": 24, "fn": 5, "tp": 744},
    "GRU": {"acc": 0.9966, "bal": 0.9903, "recall": 0.9826, "precision": 0.9787, "tn": 7800, "fp": 16, "fn": 13, "tp": 736},
}
TEST_SUBJECTS = {"SA06", "SA08", "SA11", "SA13", "SA17", "SE10", "SE11", "SE14"}
VAL_SUBJECTS = {"SA10", "SA12", "SA19", "SE02"}


def load_trial(activity: str, subject: str = "SA01", trial: int = 1) -> np.ndarray:
    df = pd.read_parquet(PARQUET_DIR / f"{subject}.parquet")
    rows = df[(df["activity"] == activity) & (df["trial"] == trial)]
    return rows[list(CHANNELS)].to_numpy(np.float32) * SCALE


def window_features(w: np.ndarray) -> dict[str, float]:
    """A few of the notebook's 75 features for one (WINDOW, 9) window."""
    acc = np.linalg.norm(w[:, 0:3], axis=1)
    gyr = np.linalg.norm(w[:, 3:6], axis=1)
    q = WINDOW // 4
    first, last = w[:q, 0:3].mean(0), w[-q:, 0:3].mean(0)
    cos = first @ last / (np.linalg.norm(first) * np.linalg.norm(last))
    return {
        "|a| max (g)": acc.max(),
        "|a| min (g)": acc.min(),
        "|ω| max (°/s)": gyr.max(),
        "jerk max (g/s)": (np.abs(np.diff(acc)) * SAMPLE_HZ).max(),
        "tilt (°)": np.degrees(np.arccos(np.clip(cos, -1, 1))),
    }


FALL = load_trial("F01")  # fall forward while walking, caused by a slip
SIT = load_trial("D07")  # slowly sit in a half height chair, wait, get up slowly
FALL_MAG = np.linalg.norm(FALL[:, 0:3], axis=1)
SIT_MAG = np.linalg.norm(SIT[:, 0:3], axis=1)
PEAK = int(FALL_MAG.argmax())


def label(text: str, size: int = 28, color=WHITE, **kw) -> Text:
    return Text(text, font_size=size, color=color, **kw)


class SisFallPipeline(Scene):
    def construct(self):
        self.title_card()
        self.dataset()
        self.signal_and_windows()
        self.features()
        self.split()
        self.models()
        self.results()
        self.motion()
        self.outro()

    # -- helpers ---------------------------------------------------------------

    def clear(self, run_time: float = 0.6):
        if self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=run_time)

    def heading(self, text: str) -> Text:
        h = label(text, 34, weight=SEMIBOLD).to_edge(UP, buff=0.45)
        self.play(FadeIn(h, shift=0.2 * DOWN), run_time=0.6)
        return h

    def signal_axes(self, seconds: float, y_max: float, y_step: float) -> tuple[Axes, VGroup]:
        axes = Axes(
            x_range=[0, seconds, 1],
            y_range=[0, y_max, y_step],
            x_length=12,
            y_length=4.2,
            axis_config={"color": DIM, "stroke_width": 2, "include_tip": False},
        ).shift(0.4 * DOWN)
        ticks = VGroup(
            *[label(f"{s}", 18, DIM).next_to(axes.c2p(s, 0), DOWN, 0.15) for s in range(0, int(seconds) + 1, 3)],
            *[label(f"{g}", 18, DIM).next_to(axes.c2p(0, g), LEFT, 0.15) for g in np.arange(0, y_max + 1e-9, y_step).astype(int)],
            label("time (s)", 20, DIM).next_to(axes.x_axis, DOWN, 0.5).align_to(axes.x_axis, RIGHT),
            label("|a| (g)", 20, DIM).next_to(axes.y_axis, UP, 0.15),
        )
        return axes, ticks

    def trace(self, axes: Axes, mag: np.ndarray, color, step: int = 4) -> VMobject:
        t = np.arange(len(mag))[::step] / SAMPLE_HZ
        line = VMobject(stroke_color=color, stroke_width=2.5)
        line.set_points_as_corners([axes.c2p(x, y) for x, y in zip(t, mag[::step])])
        return line

    # -- scenes ----------------------------------------------------------------

    def title_card(self):
        title = label("Detecting falls from a waist IMU", 52, weight=BOLD)
        sub = label("SisFall · HistGradientBoosting vs GRU", 30, DIM).next_to(title, DOWN, 0.4)
        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(sub, shift=0.2 * UP))
        self.wait(1)
        self.clear()

    def dataset(self):
        h = self.heading("The data: SisFall")

        # a person with the sensor clipped to their waist
        head = Circle(0.32, color=WHITE, stroke_width=4).shift(1.9 * UP)
        body = Line(1.55 * UP, 0.1 * DOWN, stroke_width=4)
        arms = VGroup(Line(1.1 * UP, 1.1 * UP + 0.7 * LEFT + 0.6 * DOWN), Line(1.1 * UP, 1.1 * UP + 0.7 * RIGHT + 0.6 * DOWN))
        legs = VGroup(Line(0.1 * DOWN, 1.8 * DOWN + 0.5 * LEFT), Line(0.1 * DOWN, 1.8 * DOWN + 0.5 * RIGHT))
        imu = RoundedRectangle(corner_radius=0.06, width=0.5, height=0.32, color=ACC_C, fill_opacity=1).move_to(0.05 * DOWN)
        person = VGroup(head, body, arms.set_stroke(width=4), legs.set_stroke(width=4), imu).shift(4 * LEFT + 0.3 * DOWN)
        self.play(Create(VGroup(head, body, arms, legs)), run_time=1.2)
        self.play(GrowFromCenter(imu))

        chips = VGroup(
            label("ADXL345  accelerometer  ±16 g", 24, ACC_C),
            label("ITG3200  gyroscope  ±2000 °/s", 24, ACC_C),
            label("MMA8451Q  accelerometer  ±8 g", 24, ACC_C),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).shift(1.6 * RIGHT + 1.6 * UP)
        arrow = Arrow(imu.get_right(), chips.get_left(), buff=0.2, color=ACC_C, stroke_width=3)
        self.play(GrowArrow(arrow), LaggedStart(*[FadeIn(c, shift=0.2 * RIGHT) for c in chips], lag_ratio=0.25))

        stats = VGroup(
            label("38 subjects · 23 adults (19–30) + 15 elderly (60–75)", 26),
            label("19 everyday activities · 15 fall types", 26),
            label("4,505 trials · 9 channels at 200 Hz", 26),
            label("15.9 million samples", 26),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(chips, DOWN, 0.7, aligned_edge=LEFT)
        self.play(LaggedStart(*[FadeIn(s, shift=0.2 * UP) for s in stats], lag_ratio=0.4), run_time=2.4)
        self.wait(1.5)
        self.clear()

    def signal_and_windows(self):
        h = self.heading("One trial: fall forward while walking (slip)")
        axes, ticks = self.signal_axes(15, 14, 2)
        line = self.trace(axes, FALL_MAG, ACC_C)
        self.play(Create(axes), FadeIn(ticks))
        self.play(Create(line), run_time=3, rate_func=linear)

        peak_pt = axes.c2p(PEAK / SAMPLE_HZ, FALL_MAG[PEAK])
        impact = VGroup(
            Dot(peak_pt, color=FALL_C, radius=0.08),
            label(f"impact  {FALL_MAG[PEAK]:.1f} g", 24, FALL_C).move_to(axes.c2p(9.3, FALL_MAG[PEAK]), aligned_edge=LEFT),
        )
        self.play(FadeIn(impact, scale=1.5))
        self.wait(0.5)

        note = label("2.56 s windows (512 samples), 50 % overlap", 26).next_to(h, DOWN, 0.3)
        self.play(FadeIn(note))
        starts = np.arange(0, len(FALL) - WINDOW + 1, STRIDE)
        boxes = VGroup()
        for i, s in enumerate(starts):
            lo, hi = axes.c2p(s / SAMPLE_HZ, 0), axes.c2p((s + WINDOW) / SAMPLE_HZ, 14)
            box = Rectangle(width=hi[0] - lo[0], height=hi[1] - lo[1], stroke_width=2, color=WHITE, fill_opacity=0.08)
            box.move_to((lo + hi) / 2).shift((0.06 if i % 2 else -0.06) * UP)
            boxes.add(box)
        self.play(LaggedStart(*[Create(b) for b in boxes], lag_ratio=0.15), run_time=2.5)

        hit = (starts <= PEAK) & (PEAK < starts + WINDOW)
        self.play(
            *[b.animate.set_style(stroke_color=FALL_C, fill_color=FALL_C, fill_opacity=0.25, stroke_width=3) for b, k in zip(boxes, hit) if k],
            *[b.animate.set_style(stroke_color=DIM, fill_opacity=0.0, stroke_opacity=0.35) for b, k in zip(boxes, hit) if not k],
        )
        legend = VGroup(
            label("contains the impact → fall", 24, FALL_C),
            label("no impact → dropped", 24, DIM),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to(axes.c2p(9.3, 10.5), aligned_edge=LEFT)
        self.play(FadeIn(legend))
        self.wait(2)

        # an everyday activity for contrast: every window is a negative
        h2 = label("Everyday activity: sit down in a chair, get up", 34, weight=SEMIBOLD).move_to(h)
        axes2, ticks2 = self.signal_axes(12, 14, 2)
        line2 = self.trace(axes2, SIT_MAG, ACC_C)
        sit_starts = np.arange(0, len(SIT) - WINDOW + 1, STRIDE)
        boxes2 = VGroup()
        for i, s in enumerate(sit_starts):
            lo, hi = axes2.c2p(s / SAMPLE_HZ, 0), axes2.c2p((s + WINDOW) / SAMPLE_HZ, 14)
            box = Rectangle(width=hi[0] - lo[0], height=hi[1] - lo[1], stroke_width=2, color=ADL_C, fill_opacity=0.12)
            boxes2.add(box.move_to((lo + hi) / 2).shift((0.06 if i % 2 else -0.06) * UP))
        legend2 = label("every window → everyday (ADL)", 24, ADL_C).next_to(axes2, UP, 0.1).align_to(axes2, RIGHT)
        self.play(
            FadeOut(VGroup(boxes, impact, legend, line)),
            ReplacementTransform(h, h2),
            ReplacementTransform(axes, axes2),
            ReplacementTransform(ticks, ticks2),
        )
        self.play(Create(line2), run_time=2, rate_func=linear)
        self.play(LaggedStart(*[FadeIn(b) for b in boxes2], lag_ratio=0.15), FadeIn(legend2), run_time=1.5)
        totals = label("40,704 windows · 3,595 falls · 37,109 everyday", 28, weight=SEMIBOLD).to_edge(DOWN, buff=0.3)
        self.play(FadeOut(legend2), Write(totals))
        self.wait(2)
        self.clear()

    def features(self):
        self.heading("HistGradientBoosting sees 75 summary features")
        recipe = VGroup(
            label("9 channels + 3 magnitudes  =  12 signals", 26),
            label("× mean, std, min, max, 10th & 90th percentile  =  72", 26),
            label("+ jerk max, jerk mean, trunk tilt  =  75", 26),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).shift(1.6 * UP)
        self.play(LaggedStart(*[FadeIn(r, shift=0.2 * RIGHT) for r in recipe], lag_ratio=0.5), run_time=2.2)

        fall_w = FALL[PEAK - WINDOW // 2 : PEAK + WINDOW // 2]
        sit_w = SIT[int(SIT_MAG.argmin()) - WINDOW // 2 :][:WINDOW]
        f, s = window_features(fall_w), window_features(sit_w)
        rows = [[label("feature", 24, DIM), label("fall window", 24, FALL_C), label("sitting window", 24, ADL_C)]]
        for name in f:
            rows.append([label(name, 24), label(f"{f[name]:,.1f}", 24, FALL_C), label(f"{s[name]:,.1f}", 24, ADL_C)])
        grid = VGroup(*[m for row in rows for m in row]).arrange_in_grid(rows=len(rows), cols=3, buff=(0.9, 0.22), col_alignments="lrr")
        grid.next_to(recipe, DOWN, 0.6)
        rule = Line(grid.get_left(), grid.get_right(), color=DIM, stroke_width=1.5).next_to(VGroup(*rows[0]), DOWN, 0.12)
        self.play(FadeIn(VGroup(*rows[0])), Create(rule))
        self.play(LaggedStart(*[FadeIn(VGroup(*r), shift=0.15 * UP) for r in rows[1:]], lag_ratio=0.3), run_time=2)
        note = label("real values from subject SA01", 20, DIM).next_to(grid, DOWN, 0.35)
        self.play(FadeIn(note))
        self.wait(2.5)
        self.clear()

    def split(self):
        self.heading("Split by person, not by window")
        ids = [f"SA{i:02d}" for i in range(1, 24)] + [f"SE{i:02d}" for i in range(1, 16)]
        cells = VGroup()
        for sid in ids:
            c = Circle(0.3, color=WHITE, stroke_width=2)
            cells.add(VGroup(c, label(sid, 14)))
        cells.arrange_in_grid(rows=4, cols=10, buff=0.22).shift(0.2 * UP)
        self.play(LaggedStart(*[FadeIn(c, scale=0.6) for c in cells], lag_ratio=0.03), run_time=1.8)

        anims = []
        for sid, c in zip(ids, cells):
            color = FALL_C if sid in TEST_SUBJECTS else GRU_C if sid in VAL_SUBJECTS else ADL_C
            anims.append(c[0].animate.set_style(stroke_color=color, fill_color=color, fill_opacity=0.35))
        legend = VGroup(
            label("train: 26 people", 22, ADL_C),
            label("GRU early stopping: 4 of them", 22, GRU_C),
            label("test: 8 new people (5 adults, 3 elderly)", 22, FALL_C),
        ).arrange(RIGHT, buff=0.55).next_to(cells, DOWN, 0.6)
        self.play(*anims, run_time=1.2)
        self.play(FadeIn(legend, shift=0.2 * UP))
        why = label("Windows from one person look alike, so the test asks: does it work on someone new?", 24, DIM)
        self.play(FadeIn(why.to_edge(DOWN, buff=0.4)))
        self.wait(2.5)
        self.clear()

    def models(self):
        self.heading("Two models, same windows")
        divider = DashedLine(2.8 * UP, 3.4 * DOWN, color=DIM)

        # HistGradientBoosting: many small trees whose scores add up
        def tree(color):
            nodes = [ORIGIN, 0.7 * DOWN + 0.55 * LEFT, 0.7 * DOWN + 0.55 * RIGHT,
                     1.4 * DOWN + 0.85 * LEFT, 1.4 * DOWN + 0.25 * LEFT, 1.4 * DOWN + 0.85 * RIGHT]  # fmt: skip
            edges = VGroup(*[Line(nodes[a], nodes[b], color=color, stroke_width=2) for a, b in [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)]])
            dots = VGroup(*[Dot(p, radius=0.09, color=color) for p in nodes])
            return VGroup(edges, dots)

        trees = VGroup(tree(HGB_C), label("+", 32), tree(HGB_C), label("+", 32), label("···", 32), label("+", 32), tree(HGB_C))
        trees.arrange(RIGHT, buff=0.25).scale(0.85)
        hgb_in = label("75 features", 24).next_to(trees, UP, 0.5)
        hgb_out = label("Σ scores → p(fall)", 24, HGB_C).next_to(trees, DOWN, 0.45)
        hgb_name = label("HistGradientBoosting", 30, HGB_C, weight=SEMIBOLD).next_to(hgb_in, UP, 0.5)
        hgb_size = VGroup(label("174 trees · 10,614 split nodes", 22, DIM), label("trains in seconds on a CPU", 22, DIM)).arrange(DOWN, buff=0.12)
        hgb = VGroup(hgb_name, hgb_in, trees, hgb_out, hgb_size.next_to(hgb_out, DOWN, 0.45)).move_to(3.55 * LEFT + 0.35 * DOWN)

        # GRU: reads the raw window step by step in both directions
        cells = VGroup(*[RoundedRectangle(corner_radius=0.08, width=0.6, height=0.45, color=GRU_C, fill_opacity=0.2) for _ in range(5)])
        cells.arrange(RIGHT, buff=0.32)
        fwd = VGroup(*[Arrow(cells[i].get_right() + 0.1 * UP, cells[i + 1].get_left() + 0.1 * UP, buff=0.04, stroke_width=2, max_tip_length_to_length_ratio=0.35, color=GRU_C) for i in range(4)])
        bwd = VGroup(*[Arrow(cells[i + 1].get_left() + 0.1 * DOWN, cells[i].get_right() + 0.1 * DOWN, buff=0.04, stroke_width=2, max_tip_length_to_length_ratio=0.35, color=GRU_C) for i in range(4)])
        xs = VGroup(*[Arrow(c.get_bottom() + 0.45 * DOWN, c.get_bottom(), buff=0.02, stroke_width=2, color=WHITE, max_tip_length_to_length_ratio=0.3) for c in cells])
        gru_in = label("raw window: 512 steps × 9 channels", 24).next_to(xs, DOWN, 0.15)
        pool = label("mean + max over time → MLP → p(fall)", 22, GRU_C).next_to(cells, UP, 0.45)
        gru_name = label("Bidirectional GRU", 30, GRU_C, weight=SEMIBOLD)
        gru_size = VGroup(label("2 layers × 64 units · 119,809 parameters", 22, DIM), label("~5 s per epoch on an RTX 3060", 22, DIM)).arrange(DOWN, buff=0.12)
        gru_body = VGroup(pool, cells, fwd, bwd, xs, gru_in)
        gru_name.next_to(gru_body, UP, 0.5)
        gru_size.next_to(gru_body, DOWN, 0.45)
        gru = VGroup(gru_name, gru_body, gru_size).move_to(3.55 * RIGHT + 0.35 * DOWN)
        # put the GRU input at the bottom, like data flowing up
        self.play(Create(divider))
        self.play(FadeIn(hgb_name), FadeIn(hgb_in))
        self.play(LaggedStart(*[Create(t) if isinstance(t, VGroup) else FadeIn(t) for t in trees], lag_ratio=0.2), run_time=2)
        self.play(FadeIn(hgb_out), FadeIn(hgb_size))
        self.play(FadeIn(gru_name), FadeIn(gru_in), LaggedStart(*[GrowArrow(x) for x in xs], lag_ratio=0.1))
        self.play(LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=0.1), LaggedStart(*[GrowArrow(a) for a in fwd], lag_ratio=0.15), run_time=1.2)
        self.play(LaggedStart(*[GrowArrow(a) for a in reversed(bwd)], lag_ratio=0.15), run_time=1)
        self.play(FadeIn(pool), FadeIn(gru_size))
        self.wait(2.5)
        self.clear()

    def results(self):
        h = self.heading("Results on the 8 unseen people (8,565 windows)")

        def confusion(name, r, color):
            vals = [[r["tn"], r["fp"]], [r["fn"], r["tp"]]]
            good = [[True, False], [False, True]]
            cells = VGroup()
            for i in range(2):
                for j in range(2):
                    sq = Square(1.25, stroke_color=DIM, stroke_width=1.5)
                    sq.set_fill(color if good[i][j] else FALL_C, opacity=0.35 if good[i][j] else 0.15 + 0.6 * min(vals[i][j] / 30, 1))
                    cells.add(VGroup(sq, label(f"{vals[i][j]:,}", 28, weight=SEMIBOLD)))
            cells.arrange_in_grid(2, 2, buff=0)
            rows = VGroup(label("true ADL", 20, DIM), label("true fall", 20, DIM))
            for k in range(2):
                rows[k].next_to(cells[2 * k], LEFT, 0.2)
            cols = VGroup(label("pred ADL", 20, DIM).next_to(cells[0], UP, 0.15), label("pred fall", 20, DIM).next_to(cells[1], UP, 0.15))
            title = label(name, 30, color, weight=SEMIBOLD).next_to(cols, UP, 0.3)
            acc = label(f"accuracy {r['acc']:.2%} · balanced {r['bal']:.2%}", 22).next_to(cells, DOWN, 0.3)
            return VGroup(title, cols, rows, cells, acc)

        hgb = confusion("HistGradientBoosting", RESULTS["HGB"], HGB_C).move_to(3.4 * LEFT + 0.4 * DOWN)
        gru = confusion("GRU", RESULTS["GRU"], GRU_C).move_to(3.4 * RIGHT + 0.4 * DOWN)
        self.play(FadeIn(hgb, shift=0.3 * RIGHT), FadeIn(gru, shift=0.3 * LEFT), run_time=1.2)
        self.wait(1)

        # point at the two cells that differ: missed falls and false alarms
        h_fn, h_fp = hgb[3][2], hgb[3][1]
        g_fn, g_fp = gru[3][2], gru[3][1]
        ring = lambda m, c: SurroundingRectangle(m, color=c, buff=0.04, stroke_width=5)  # noqa: E731
        missed = label("missed falls: HGB 5 vs GRU 13", 28, HGB_C).to_edge(DOWN, buff=0.35)
        self.play(Create(ring(h_fn, HGB_C)), Create(ring(g_fn, HGB_C)), Write(missed))
        self.wait(1.5)
        alarms = label("false alarms: HGB 24 vs GRU 16", 28, GRU_C).to_edge(DOWN, buff=0.35)
        self.play(Create(ring(h_fp, GRU_C)), Create(ring(g_fp, GRU_C)), ReplacementTransform(missed, alarms))
        self.wait(1.5)
        verdict = label("Same accuracy. HGB catches more falls; the GRU raises fewer false alarms.", 26, weight=SEMIBOLD)
        self.play(ReplacementTransform(alarms, verdict.to_edge(DOWN, buff=0.35)))
        self.wait(2.5)
        self.clear()

    def motion(self):
        self.heading("Where did the waist go? (reconstructed from accel + gyro)")
        pos, _, _ = reconstruct(FALL[:, 6:9], FALL[:, 3:6])
        # side view: horizontal = main direction of travel, vertical = height
        xy = pos[:, :2] - pos[:, :2].mean(0)
        direction = np.linalg.svd(xy, full_matrices=False)[2][0]
        horiz = xy @ direction
        horiz -= (horiz.min() + horiz.max()) / 2
        height = pos[:, 2] + STANDING

        room = Axes(x_range=[-1.0, 1.0, 0.5], y_range=[0, 1.0, 0.5], x_length=5.0, y_length=4.0, tips=False,
                    axis_config={"color": DIM, "stroke_opacity": 0}).shift(3.3 * LEFT + 0.6 * DOWN)  # fmt: skip
        floor = Line(room.c2p(-1.0, 0), room.c2p(1.0, 0), color=WHITE, stroke_width=3)
        stand = DashedLine(room.c2p(-1.0, STANDING), room.c2p(1.0, STANDING), color=DIM, stroke_width=1.5)
        stand_label = label("standing waist height", 18, DIM).next_to(stand, UP, 0.08).align_to(stand, LEFT)

        axes, ticks = self.signal_axes(15, 14, 2)
        plot = VGroup(axes, ticks, self.trace(axes, FALL_MAG, ACC_C)).scale(0.5).shift(3.3 * RIGHT + 0.3 * DOWN)

        t = ValueTracker(2.0)
        idx = lambda: min(int(t.get_value() * SAMPLE_HZ), len(FALL) - 1)  # noqa: E731

        def sphere_color():
            a = FALL_MAG[idx()]
            return interpolate_color(ManimColor(HGB_C), ManimColor(FALL_C), np.clip((a - 1) / 3, 0, 1)) if a > 1 else HGB_C

        sphere = always_redraw(lambda: Dot(room.c2p(horiz[idx()], height[idx()]), radius=0.16, color=sphere_color()))
        trail = TracedPath(sphere.get_center, stroke_color=ACC_C, stroke_width=5, dissipating_time=4.0)
        start = Dot(room.c2p(horiz[0], height[0]), radius=0.16, color=DIM, fill_opacity=0.35)
        cursor = always_redraw(lambda: Line(axes.c2p(t.get_value(), 0), axes.c2p(t.get_value(), 14), color=WHITE, stroke_width=2))
        clock = always_redraw(lambda: label(f"t = {t.get_value():4.1f} s", 22, DIM).next_to(plot, UP, 0.25))

        self.play(Create(floor), Create(stand), FadeIn(stand_label), FadeIn(plot))
        self.add(start, trail, sphere, cursor, clock)
        self.play(t.animate.set_value(14.9), run_time=12.9, rate_func=linear)  # real time
        caveat = label("Double-integrated position drifts, so the drop is underestimated — an estimate, not ground truth.", 22, DIM)
        self.play(FadeIn(caveat.to_edge(DOWN, buff=0.4)))
        self.wait(2)
        self.clear()

    def outro(self):
        lines = VGroup(
            label("SisFall → 2.56 s windows → features / raw signal → HGB & GRU", 30),
            label("99.66 % accuracy on people the models never saw", 30, ADL_C),
            label("Interactive 3D replay of every subject with Rerun", 30, ACC_C),
            label("github.com/skynse/ysafetyml", 28, DIM),
        ).arrange(DOWN, buff=0.45)
        self.play(LaggedStart(*[FadeIn(l, shift=0.2 * UP) for l in lines], lag_ratio=0.4), run_time=2.5)
        self.wait(3)
        self.play(FadeOut(lines))
