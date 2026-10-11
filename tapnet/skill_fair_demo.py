# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0
#
# TAP-VISION - Skill Fair Edition
# Real-Time Any-Point Tracking & Motion Visualization

import time

import cv2
import numpy as np
from tapnet.torch import tapir_model
import torch
import torch.nn.functional as F
import tree


NUM_POINTS = 8

WINDOW_NAME = "TAP-VISION | Real-Time Point Tracking"

POINT_COLORS = [
    (255, 80, 80),
    (80, 255, 120),
    (80, 180, 255),
    (255, 100, 255),
    (255, 220, 80),
    (80, 255, 255),
    (180, 100, 255),
    (255, 255, 255),
]

MAX_TRAIL_LENGTH = 45


def preprocess_frames(frames):
  frames = frames.float()
  frames = frames / 255 * 2 - 1
  return frames


def online_model_init(frames, points):
  frames = preprocess_frames(frames)

  feature_grids = model.get_feature_grids(
      frames,
      is_training=False,
  )

  features = model.get_query_features(
      frames,
      is_training=False,
      query_points=points,
      feature_grids=feature_grids,
  )

  return features


def postprocess_occlusions(occlusions, expected_dist):
  visibles = (
      (1 - F.sigmoid(occlusions))
      * (1 - F.sigmoid(expected_dist))
      > 0.5
  )
  return visibles


def online_model_predict(frames, features, causal_context):
  frames = preprocess_frames(frames)

  feature_grids = model.get_feature_grids(
      frames,
      is_training=False,
  )

  trajectories = model.estimate_trajectories(
      frames.shape[-3:-1],
      is_training=False,
      feature_grids=feature_grids,
      query_features=features,
      query_points_in_video=None,
      query_chunk_size=64,
      causal_context=causal_context,
      get_causal_context=True,
  )

  causal_context = trajectories["causal_context"]

  del trajectories["causal_context"]

  tracks = trajectories["tracks"][-1]
  occlusions = trajectories["occlusion"][-1]
  uncertainty = trajectories["expected_dist"][-1]

  visibles = postprocess_occlusions(
      occlusions,
      uncertainty,
  )

  return tracks, visibles, causal_context


def get_frame(video_capture):
  r_val, image = video_capture.read()

  if image is None:
    return False, image

  trunc = abs(image.shape[1] - image.shape[0]) // 2

  if image.shape[1] > image.shape[0]:
    image = image[:, trunc:-trunc]

  elif image.shape[1] < image.shape[0]:
    image = image[trunc:-trunc]

  return r_val, image


def draw_panel(frame, fps, active_points, tracking):
  height, width = frame.shape[:2]

  panel_width = 250

  panel = np.zeros(
      (height, panel_width, 3),
      dtype=np.uint8,
  )

  panel[:] = (25, 25, 30)

  cv2.putText(
      panel,
      "TAP-VISION",
      (18, 35),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.75,
      (255, 255, 255),
      2,
  )

  cv2.putText(
      panel,
      "Real-Time Tracking",
      (18, 62),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.48,
      (180, 180, 180),
      1,
  )

  cv2.line(
      panel,
      (18, 78),
      (panel_width - 18, 78),
      (80, 80, 80),
      1,
  )

  cv2.putText(
      panel,
      "STATUS",
      (18, 110),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.42,
      (150, 150, 150),
      1,
  )

  status_text = "TRACKING" if tracking else "READY"
  status_color = (80, 255, 120) if tracking else (100, 200, 255)

  cv2.putText(
      panel,
      status_text,
      (18, 137),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.65,
      status_color,
      2,
  )

  cv2.putText(
      panel,
      f"Points: {active_points}/{NUM_POINTS}",
      (18, 180),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.55,
      (230, 230, 230),
      1,
  )

  cv2.putText(
      panel,
      f"FPS: {fps:.1f}",
      (18, 210),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.55,
      (230, 230, 230),
      1,
  )

  cv2.line(
      panel,
      (18, 235),
      (panel_width - 18, 235),
      (80, 80, 80),
      1,
  )

  cv2.putText(
      panel,
      "HOW IT WORKS",
      (18, 267),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.42,
      (150, 150, 150),
      1,
  )

  instructions = [
      "1. Click a point",
      "2. Move the object",
      "3. Watch it track",
  ]

  y = 295

  for line in instructions:
    cv2.putText(
        panel,
        line,
        (18, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (220, 220, 220),
        1,
    )
    y += 28

  cv2.putText(
      panel,
      "CONTROLS",
      (18, 390),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.42,
      (150, 150, 150),
      1,
  )

  controls = [
      "Click  Add point",
      "R      Reset",
      "ESC    Exit",
  ]

  y = 418

  for line in controls:
    cv2.putText(
        panel,
        line,
        (18, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (220, 220, 220),
        1,
    )
    y += 27

  cv2.putText(
      panel,
      "Powered by TAPIR",
      (18, height - 20),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.42,
      (120, 120, 120),
      1,
  )

  return np.hstack((frame, panel))


print("==============================================")
print("             TAP-VISION")
print("     Real-Time Any-Point Tracking")
print("==============================================")
print()
print("Starting TAPIR...")
print("Keep the laptop connected to power.")
print()

if torch.cuda.is_available():
  device = torch.device("cuda")
  print("GPU acceleration: ENABLED")
  print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
  device = torch.device("cpu")
  print("GPU acceleration: DISABLED")

torch.set_default_device(device)

print()
print("Creating model...")

model = tapir_model.TAPIR(
    pyramid_level=1,
    use_casual_conv=True,
)

print("Loading checkpoint...")

model.load_state_dict(
    torch.load(
        "tapnet/checkpoints/causal_bootstapir_checkpoint.pt"
    )
)

model = model.to(device)
model = model.eval()

torch.set_grad_enabled(False)

print("Initializing camera...")

vc = cv2.VideoCapture(0)

vc.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    240,
)

if not vc.isOpened():
  raise ValueError("Unable to open camera.")

rval, frame = get_frame(vc)

if not rval or frame is None:
  raise ValueError("Unable to read camera frame.")

pos = tuple()
query_frame = True

have_point = [False] * NUM_POINTS

query_points = torch.zeros(
    [NUM_POINTS, 3],
    dtype=torch.float32,
)

query_points = query_points.to(device)

frame_tensor = torch.tensor(frame).to(device)

query_features = online_model_init(
    frames=frame_tensor[None, None],
    points=query_points[None, :],
)

causal_state = model.construct_initial_causal_state(
    NUM_POINTS,
    len(query_features.resolutions) - 1,
)

causal_state = tree.map_structure(
    lambda x: x.to(device),
    causal_state,
)

prediction, visible, causal_state = online_model_predict(
    frames=frame_tensor[None, None],
    features=query_features,
    causal_context=causal_state,
)

next_query_idx = 0
last_click_time = 0

trail_history = [
    [] for _ in range(NUM_POINTS)
]


def reset_tracking():
  global pos
  global query_frame
  global next_query_idx
  global have_point
  global trail_history

  pos = tuple()
  query_frame = False
  next_query_idx = 0

  have_point = [False] * NUM_POINTS

  trail_history = [
      [] for _ in range(NUM_POINTS)
  ]


def mouse_click(event, x, y, flags, param):
  del flags, param

  global pos
  global query_frame
  global last_click_time

  if (time.time() - last_click_time) < 0.5:
    return

  if event == cv2.EVENT_LBUTTONDOWN:

    pos = (
        y,
        frame.shape[1] - x,
    )

    query_frame = True

    last_click_time = time.time()


cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL,
)

cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_click,
)

t = time.time()
step_counter = 0
current_fps = 0.0

print()
print("TAP-VISION READY")
print("Click anywhere in the camera feed to track a point.")
print("Press R to reset all points.")
print("Press ESC to exit.")
print()

while rval:

  rval, frame = get_frame(vc)

  if not rval or frame is None:
    break

  numpy_frame = frame.copy()

  if query_frame:

    query_points = np.array(
        (0,) + pos,
        dtype=np.float32,
    )

    frame_tensor = torch.tensor(frame).to(device)

    query_points = torch.tensor(
        query_points
    ).to(device)

    init_query_features = online_model_init(
        frames=frame_tensor[None, None],
        points=query_points[None, None],
    )

    query_frame = False

    query_features, causal_state = (
        model.update_query_features(
            query_features=query_features,
            new_query_features=init_query_features,
            idx_to_update=np.array(
                [next_query_idx]
            ),
            causal_state=causal_state,
        )
    )

    trail_history[next_query_idx] = []

    have_point[next_query_idx] = True

    next_query_idx = (
        next_query_idx + 1
    ) % NUM_POINTS

  if pos:

    frame_tensor = torch.tensor(frame).to(device)

    track, visible, causal_state = (
        online_model_predict(
            frames=frame_tensor[None, None],
            features=query_features,
            causal_context=causal_state,
        )
    )

    track = np.round(
        track.cpu().numpy()
    )

    visible = visible.cpu().numpy()

    for i in range(NUM_POINTS):

      if (
          visible[0, i, 0]
          and have_point[i]
      ):

        x = int(
            track[0, i, 0, 0]
        )

        y = int(
            track[0, i, 0, 1]
        )

        color = POINT_COLORS[i]

        trail_history[i].append(
            (x, y)
        )

        if (
            len(trail_history[i])
            > MAX_TRAIL_LENGTH
        ):
          trail_history[i].pop(0)

        for j in range(
            1,
            len(trail_history[i]),
        ):

          cv2.line(
              numpy_frame,
              trail_history[i][j - 1],
              trail_history[i][j],
              color,
              2,
          )

        cv2.circle(
            numpy_frame,
            (x, y),
            8,
            color,
            -1,
        )

        cv2.circle(
            numpy_frame,
            (x, y),
            11,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            numpy_frame,
            f"P{i + 1}",
            (x + 12, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

  step_counter += 1

  elapsed = time.time() - t

  if elapsed >= 1.0:

    current_fps = (
        step_counter / elapsed
    )

    step_counter = 0
    t = time.time()

  active_points = sum(have_point)

  display_frame = draw_panel(
      numpy_frame[:, ::-1],
      current_fps,
      active_points,
      active_points > 0,
  )

  cv2.imshow(
      WINDOW_NAME,
      display_frame,
  )

  key = cv2.waitKey(1) & 0xFF

  if key == 27:
    break

  if key == ord("r"):
    reset_tracking()

cv2.destroyWindow(WINDOW_NAME)
vc.release()

print()
print("TAP-VISION closed.")