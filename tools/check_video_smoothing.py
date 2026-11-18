"""Check whether FaceLandmarker VIDEO mode applies the One Euro landmark smoothing.

Runs FaceLandmarker in IMAGE and VIDEO mode (num_faces=1) on a video and compares the
frame-to-frame jitter of the two iris centres. With the smoothing filter active,
VIDEO jitter is several times lower than IMAGE jitter; with it disabled both are
of the same order (VIDEO still tracks, so it can stay a bit lower).

Usage (do NOT run it from the repository root: the local `mediapipe/` folder would
shadow the installed wheel):
    python tools/check_video_smoothing.py --video v.avi --task face_landmarker.task
Needs: opencv-python(-headless), numpy.
"""
import argparse
import time

import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

IRIS = {"iris_a": 468, "iris_b": 473}


def run(model, mode, frames, fps):
  lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
      base_options=mpt.BaseOptions(model_asset_buffer=model), running_mode=mode, num_faces=1))
  out = {k: [] for k in IRIS}
  t_total = 0.0
  for i, img in enumerate(frames):
    t0 = time.perf_counter()
    if mode == vision.RunningMode.VIDEO:
      r = lm.detect_for_video(img, int(round(i * 1000.0 / fps)))
    else:
      r = lm.detect(img)
    t_total += time.perf_counter() - t0
    for k, idx in IRIS.items():
      if r.face_landmarks:
        p = r.face_landmarks[0][idx]
        out[k].append((p.x * img.width, p.y * img.height))
      else:
        out[k].append((np.nan, np.nan))
  return {k: np.array(v) for k, v in out.items()}, 1000 * t_total / len(frames)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--video", required=True)
  ap.add_argument("--task", required=True)
  ap.add_argument("--max-frames", type=int, default=2000)
  a = ap.parse_args()
  cap = cv.VideoCapture(a.video)
  fps = cap.get(cv.CAP_PROP_FPS) or 30.0
  frames = []
  while len(frames) < a.max_frames:
    ok, bgr = cap.read()
    if not ok:
      break
    frames.append(mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=np.ascontiguousarray(cv.cvtColor(bgr, cv.COLOR_BGR2RGB))))
  model = open(a.task, "rb").read()  # buffer: avoids Windows absolute-path issue
  img, ms_i = run(model, vision.RunningMode.IMAGE, frames, fps)
  vid, ms_v = run(model, vision.RunningMode.VIDEO, frames, fps)
  ratios = []
  print(f"{len(frames)} frames @ {fps:g} fps | ms/frame: IMAGE {ms_i:.2f}  VIDEO {ms_v:.2f}")
  print(f"{'point/axis':14}{'IMAGE jitter':>14}{'VIDEO jitter':>14}{'VIDEO/IMAGE':>13}")
  for k in IRIS:
    for ax, name in ((0, "x"), (1, "y")):
      ji = np.nanstd(np.diff(img[k][:, ax]))
      jv = np.nanstd(np.diff(vid[k][:, ax]))
      ratios.append(jv / ji)
      print(f"{k + ' ' + name:14}{ji:14.4f}{jv:14.4f}{jv / ji:13.2f}")
  m = float(np.mean(ratios))
  print(f"mean VIDEO/IMAGE jitter ratio: {m:.2f}")
  print("=> smoothing filter looks ACTIVE (ratio < 0.5)" if m < 0.5
        else "=> smoothing filter looks DISABLED (ratio >= 0.5)")


if __name__ == "__main__":
  main()
