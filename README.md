# Mosquito killer project

Computer-vision mosquito pointer: detect a mosquito with a camera and point a
beam at it so a human can find and swat it.

## Web simulator (`simulator/`)

Browser-based simulator that runs the real detection pipeline on your MacBook
camera and renders a **virtual** laser/flashlight beam on screen — beam
direction and width are drawn as if an emitter at the chosen screen corner were
pointing at the target.

### Run

```sh
python3 -m http.server 8788
# then open http://localhost:8788/simulator/
```

(Any static server works; `getUserMedia` needs localhost or https.)

### How it works

1. Camera frames are downscaled to 320 px wide grayscale.
2. Running-average background subtraction + threshold → binary motion mask.
3. Connected-component blob detection, filtered by min/max size.
4. Tracker with lock / coast / search states and constant-velocity smoothing.
5. Overlay renders the beam cone (origin → target, widening with throw
   distance), a landing spot, and a pulsing targeting ring around the bug.

This mirrors the planned hardware: steps 1–4 are what the vision host (Pi 5)
runs; step 5 is replaced by the galvo/servo-steered real laser.

### Controls

- **Camera** — pick webcam (Continuity Camera works; disable Center Stage for widest FOV).
- **Beam style** — laser pointer (thin, green) or flashlight (wide, warm).
- **Beam origin** — which screen corner/edge the virtual emitter sits at.
- **Sensitivity** — motion threshold (higher = detects fainter motion).
- **Beam width** — divergence of the cone.
- **Min/Max size** — blob area filter in processing pixels (keeps small bugs, rejects people).
- **Demo bug** — synthetic mosquito injected into the detector for testing.
- **Debug view** — shows the motion mask, insect candidates (green) and
  rejected large objects (red).
- Click video = manual target, double-click = clear.

### Human rejection & snapshots

- Frames where a large fraction of pixels are moving (someone walking by,
  lighting change) are ignored entirely; blobs adjacent to any large moving
  region (hands, hair fringes) are rejected as fragments.
- A per-pixel **skin mask** (normalized-RGB) marks faces and hands. Any
  candidate whose neighborhood is mostly skin is rejected — this kills
  blinking eyes, nostrils and mouth, which flicker as insect-sized motion
  blobs but sit inside a skin region. Debug view tints skin blue.
- A lock must **travel** (net displacement > a few px) to gain confidence; an
  in-place flicker is starved of confidence and its lock expires, so a real
  moving target elsewhere can be acquired instead.
- A heuristic **bug confidence** accumulates while a lock is held: insect-sized
  area + insect-like speed + erratic heading changes, sustained over ~2-3 s.
  It is shown in the HUD. When it crosses **99%**, a full-resolution snapshot
  is captured automatically with the detection marked (ring + crosshair +
  timestamp + confidence) and added to the Detections gallery; click a
  thumbnail to download the JPEG. Note: this is a motion/shape heuristic, not
  a species classifier — it means "definitely a small flying insect".
