"""Compare eye movement (x / y) from MediaPipe legacy FaceMesh and FaceLandmarker (VIDEO, IMAGE).

Needs a mediapipe build that still ships both `mp.solutions` and `mp.tasks`
(PyPI: mediapipe<=0.10.21), plus opencv-python(-headless), numpy and matplotlib.

Usage (run it from anywhere, but NOT with a script placed next to the `mediapipe/` source folder):
    python compare_eyes.py --video ../enzo_vergence_remobi_1.avi --task ../face_landmarker.task
    python compare_eyes.py --video v.avi --task f.task --zoom 10 14 --show

Eye position = vector from the iris centre to the orbit reference, projected on the head axes
(horizontal: orbit contour, vertical: eyebrows), as described in the project's geometry step.
Head gains = 1 and no Z-EMA, so only MediaPipe's own differences show up.
"""
import argparse
import csv
import os

import cv2 as cv
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

IRIS_L, IRIS_R = [474, 475, 476, 477], [469, 470, 471, 472]
CONT_L = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
CONT_R = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
BROW_L = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]
BROW_R = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

MODES = ["legacy", "task_video", "task_video_nofilter", "task_image"]
LABELS = {"legacy": "Legacy (FaceMesh, refine)", "task_video": "Tasks VIDEO", "task_video_nofilter": "Tasks VIDEO (num_faces=2, no smoothing)", "task_image": "Tasks IMAGE"}
COLORS = {"legacy": "tab:blue", "task_video": "tab:red", "task_video_nofilter": "tab:orange", "task_image": "tab:green"}
SERIES = ["eye_left_x", "eye_left_y", "eye_right_x", "eye_right_y"]
TITLES = {"eye_left_x": "Left eye - X", "eye_left_y": "Left eye - Y",
          "eye_right_x": "Right eye - X", "eye_right_y": "Right eye - Y"}


def to_px(lms, w, h):
    return np.array([[l.x * w, l.y * h, l.z * w] for l in lms], dtype=np.float64)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def eye_positions(lms, w, h):
    p = to_px(lms, w, h)
    hh, vv = unit(p[263] - p[33]), unit(p[2] - p[152])
    out = {}
    for side, ir, ct, br in (("left", IRIS_L, CONT_L, BROW_L), ("right", IRIS_R, CONT_R, BROW_R)):
        pts = p[ir]
        (cx, cy), _ = cv.minEnclosingCircle(pts[:, :2].astype(np.float32))
        c = np.array([cx, cy, pts[:, 2].mean()])
        out[f"eye_{side}_x"] = float(np.dot(c - p[ct].mean(0), hh))
        out[f"eye_{side}_y"] = float(np.dot(c - p[br].mean(0), vv))
    return out


def analyse(video, task, max_frames):
    cap = cv.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    legacy = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                             min_detection_confidence=0.5, min_tracking_confidence=0.5)
    model = open(task, "rb").read()  # buffer: avoids Windows absolute-path issue

    def make(mode, num_faces=1):
        return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_buffer=model), running_mode=mode, num_faces=num_faces,
            min_face_detection_confidence=0.5, min_face_presence_confidence=0.5, min_tracking_confidence=0.5))

    t_video, t_image = make(vision.RunningMode.VIDEO), make(vision.RunningMode.IMAGE)
    # num_faces=2: the graph only enables smooth_landmarks when num_faces == 1, so no One Euro filter here
    t_video_nf = make(vision.RunningMode.VIDEO, num_faces=2)
    data = {m: {k: [] for k in SERIES} for m in MODES}
    nan = {k: np.nan for k in SERIES}

    i = 0
    while max_frames is None or i < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = np.ascontiguousarray(cv.cvtColor(bgr, cv.COLOR_BGR2RGB))
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        r = legacy.process(rgb)
        res = {"legacy": eye_positions(r.multi_face_landmarks[0].landmark, w, h) if r.multi_face_landmarks else nan}
        r = t_video.detect_for_video(img, int(round(i * 1000.0 / fps)))
        res["task_video"] = eye_positions(r.face_landmarks[0], w, h) if r.face_landmarks else nan
        r = t_video_nf.detect_for_video(img, int(round(i * 1000.0 / fps)))
        res["task_video_nofilter"] = eye_positions(r.face_landmarks[0], w, h) if r.face_landmarks else nan
        r = t_image.detect(img)
        res["task_image"] = eye_positions(r.face_landmarks[0], w, h) if r.face_landmarks else nan

        for m in MODES:
            for k in SERIES:
                data[m][k].append(res[m][k])
        i += 1
        if i % 500 == 0:
            print(f"  {i} frames...", flush=True)
    cap.release()
    t = np.arange(i) / fps
    return t, {m: {k: np.array(v) for k, v in d.items()} for m, d in data.items()}, fps


def plot(t, data, out_png, title, xlim=None, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for ax, k in zip(axes, SERIES):
        for m in MODES:
            ax.plot(t, data[m][k], color=COLORS[m], lw=0.9, alpha=0.85, label=LABELS[m])
        ax.set_ylabel(f"{TITLES[k]}\n(px)")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", ncol=3)
    axes[-1].set_xlabel("time (s)")
    if xlim:
        axes[-1].set_xlim(*xlim)
        # rescale y to the visible window
        for ax, k in zip(axes, SERIES):
            sel = (t >= xlim[0]) & (t <= xlim[1])
            vals = np.concatenate([data[m][k][sel] for m in MODES])
            vals = vals[~np.isnan(vals)]
            if vals.size:
                pad = 0.05 * (vals.max() - vals.min() + 1e-9)
                ax.set_ylim(vals.min() - pad, vals.max() + pad)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print("saved", out_png)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--task", required=True, help="face_landmarker.task")
    ap.add_argument("--out-dir", default=".", help="where to write PNGs and CSV")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--zoom", type=float, nargs=2, metavar=("START_S", "END_S"), default=None,
                    help="also save a zoomed figure on this time window (seconds)")
    ap.add_argument("--show", action="store_true", help="open interactive matplotlib windows")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    print("analysing", a.video)
    t, data, fps = analyse(a.video, a.task, a.max_frames)
    print(f"{len(t)} frames @ {fps:g} fps")

    csv_path = os.path.join(a.out_dir, "eye_curves.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "time_s"] + [f"{m}:{k}" for m in MODES for k in SERIES])
        for j in range(len(t)):
            wr.writerow([j, f"{t[j]:.4f}"] + [data[m][k][j] for m in MODES for k in SERIES])
    print("saved", csv_path)

    name = os.path.basename(a.video)
    plot(t, data, os.path.join(a.out_dir, "eye_curves_full.png"), f"Eye movement - {name} (full)", show=a.show)
    if a.zoom:
        plot(t, data, os.path.join(a.out_dir, "eye_curves_zoom.png"),
             f"Eye movement - {name} ({a.zoom[0]:g}-{a.zoom[1]:g} s)", xlim=tuple(a.zoom), show=a.show)
    if a.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
