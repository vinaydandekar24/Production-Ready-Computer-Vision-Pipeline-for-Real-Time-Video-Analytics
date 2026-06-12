"""
local_speed_runner.py
─────────────────────
Runs SpeedEstimator._process_frame directly on your local CPU machine
and pipes annotated frames to a C++ RTSP streamer (live_streaming).

No command-line arguments needed — edit the RUN CONFIG block below,
then just run:   python local_speed_runner.py

What this file does differently from the server version (main_k.py):
  • No GPU required  — forces CPU-only inference everywhere
  • Annotated frames are written to a named pipe (FIFO) so the companion
    C++ program (live_streaming) can forward them as RTSP via MediaMTX
  • No cv2.imshow preview by default (headless pipe mode)
  • Saves annotated output to a video file alongside the input (optional)
  • Still writes speed estimates to MongoDB (same DB as the server)
  • Still loads calibration from MongoDB (same collection)

Requirements (install once):
  pip install ultralytics opencv-python pymongo supervision
  # supervision is optional but gives better tracking (ByteTrack)

How it connects to live_streaming (C++):
  1. This script creates a named pipe at PIPE_PATH (default: /tmp/speed_pipe)
     and writes raw BGR frames in the format: width(4B LE) | height(4B LE) | pixels
  2. The C++ program reads from that pipe as its RTSP_CAMERA_URL using
     ffmpeg's rawvideo demuxer and forwards the stream to MediaMTX as
     rtsp://<server>:8554/mycamera
  3. Start order: run this script first, then live_streaming — or start
     both; this script will block on the pipe open until the C++ reader connects.
"""

import logging
import os
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, Optional, Set

# ── Force CPU before any torch import ────────────────────────────────────────
# This makes CUDA invisible even if drivers exist — must happen before import.
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # hide all GPUs
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"

import cv2
import numpy as np
import torch
import struct
import stat

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("local_runner")
logger.info("Running on CPU (GPU disabled for local mode)")

# ── Optional deps ─────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    logger.error("ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

try:
    import supervision as sv
    HAS_SV = True
except ImportError:
    HAS_SV = False
    logger.warning("supervision not installed — using SimpleTracker fallback")

from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError

# ─────────────────────────────────────────────────────────────────────────────
# Config  (mirrors main_k.py — edit to match your deployment)
# ─────────────────────────────────────────────────────────────────────────────

MONGO_URI        = "mongodb+srv://speeddetection:29d0wwwu8J8mjlql@cluster0.toy45ng.mongodb.net/"
DB_NAME          = "vehicle"
CALIBRATION_COL  = "calibration"
SPEED_EST_COL    = "speedEstimates"

YOLO_MODEL_PATH  = "yolov8n_openvino_model"   # change to yolov8n.pt if no openvino model

CONF_THRESHOLD   = 0.4
NMS_IOU          = 0.45
BLUR_THRESHOLD   = 100.0
FRAME_STEP       = 1          # process every Nth frame (raise to 2/3 on slow CPUs)
MIN_SPEED_MPS    = 1.0

# Vehicle class IDs (COCO)
VEHICLE_CLASSES  = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

DB_INSERT_COOLDOWN_S = 3.0
STALE_TRACK_FRAMES   = 60

PREVIEW_MAX_W    = 1280       # resize preview window if wider than this
SAVE_OUTPUT      = False      # default; overridden by --save-output flag

# ── Pipe / RTSP output ────────────────────────────────────────────────────────
# Path of the named pipe (FIFO) that live_streaming.cpp reads from.
# The C++ program treats this as its input "camera" source via FFmpeg rawvideo.
PIPE_PATH        = "/tmp/speed_pipe"

# Frame format written into the pipe.
# 4 bytes (little-endian uint32) width + 4 bytes height + raw BGR pixels.
# The C++ / FFmpeg side must use:  -f rawvideo -pix_fmt bgr24 -video_size WxH
PIPE_FRAME_HEADER_FMT = "<II"   # little-endian unsigned int, unsigned int


# ─────────────────────────────────────────────────────────────────────────────
# MongoDB helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_db():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
    return client, client[DB_NAME]


# Path to local calibration JSON file.
# The file can contain calibration for ONE camera (a plain dict)
# or MULTIPLE cameras (a dict keyed by camera_id, or a list of dicts).
LOCAL_CALIBRATION_FILE = "calibration.json"


def fetch_calibration(db, camera_id: str) -> Optional[dict]:
    """
    Load calibration in this priority order:
      1. Local JSON file  (LOCAL_CALIBRATION_FILE) — tried first, no network needed
      2. MongoDB          — fallback if the file is missing or doesn't have this camera_id
    """
    import json

    # ── 1. Try local file ────────────────────────────────────────────────
    if os.path.isfile(LOCAL_CALIBRATION_FILE):
        try:
            with open(LOCAL_CALIBRATION_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)

            doc = None
            if isinstance(raw, list):
                # List of dicts — match by camera_id key if present, else use first entry
                for entry in raw:
                    if entry.get("camera_id") == camera_id:
                        doc = {k: v for k, v in entry.items() if k not in ("_id", "camera_id")}
                        break
                if doc is None and raw:
                    # No camera_id match — use first entry as-is
                    first = raw[0]
                    doc = {k: v for k, v in first.items() if k not in ("_id", "camera_id")}
                    logger.warning(
                        f"[{camera_id}] No camera_id match in list — using first entry"
                    )
            elif isinstance(raw, dict):
                if camera_id in raw:
                    # Multi-camera keyed dict  {"CAM_001": {...}, ...}
                    doc = {k: v for k, v in raw[camera_id].items() if k not in ("_id", "camera_id")}
                elif "camera_id" in raw and raw["camera_id"] == camera_id:
                    # Single-camera dict with matching camera_id field
                    doc = {k: v for k, v in raw.items() if k not in ("_id", "camera_id")}
                elif "source_points" in raw:
                    # ✓ Plain calibration dict with no camera_id at all — use directly
                    doc = {k: v for k, v in raw.items() if k not in ("_id", "camera_id")}
                    logger.info(
                        f"[{camera_id}] calibration.json has no camera_id field — loaded directly"
                    )

            if doc is not None:
                logger.info(
                    f"[{camera_id}] Calibration loaded from local file: {LOCAL_CALIBRATION_FILE}"
                )
                return doc
            else:
                logger.warning(
                    f"[{camera_id}] Could not match camera in {LOCAL_CALIBRATION_FILE} — trying MongoDB"
                )
        except Exception as exc:
            logger.warning(
                f"[{camera_id}] Could not read {LOCAL_CALIBRATION_FILE}: {exc} — trying MongoDB"
            )
    else:
        logger.info(
            f"[{camera_id}] {LOCAL_CALIBRATION_FILE} not found — trying MongoDB"
        )

    # ── 2. Fallback: MongoDB ─────────────────────────────────────────────
    doc = db[CALIBRATION_COL].find_one(
        {"camera_id": camera_id},
        {"_id": 0, "camera_id": 0},
    )
    if doc:
        logger.info(f"[{camera_id}] Calibration loaded from MongoDB")
    else:
        logger.error(f"[{camera_id}] No calibration found in '{CALIBRATION_COL}' (MongoDB)")
    return doc


def insert_speed_estimate(db, record: dict):
    try:
        db[SPEED_EST_COL].insert_one(record)
        logger.debug(
            f"DB insert: tid={record['tracker_id']} "
            f"{record['speed_kmh']} km/h"
        )
    except PyMongoError as exc:
        logger.error(f"MongoDB insert error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_blurry(frame: np.ndarray, threshold: float = BLUR_THRESHOLD) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


def _iou(a, b) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / u if u > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Simple IoU tracker fallback
# ─────────────────────────────────────────────────────────────────────────────

class SimpleTracker:
    def __init__(self, iou_thr: float = 0.3, max_lost: int = 15):
        self.iou_thr  = iou_thr
        self.max_lost = max_lost
        self._nxt     = 1
        self._tracks: dict = {}

    def update(self, boxes: list, cls_ids: list) -> list:
        unmatched = list(range(len(boxes)))
        matched: dict = {}
        for tid, tr in self._tracks.items():
            best, bi = 0.0, -1
            for i in unmatched:
                v = _iou(tr["box"], boxes[i])
                if v > best:
                    best, bi = v, i
            if best >= self.iou_thr:
                matched[tid] = bi
                unmatched.remove(bi)
        for tid, idx in matched.items():
            self._tracks[tid].update(
                {"box": boxes[idx], "lost": 0, "cls": cls_ids[idx]}
            )
        for tid in list(self._tracks):
            if tid not in matched:
                self._tracks[tid]["lost"] += 1
                if self._tracks[tid]["lost"] > self.max_lost:
                    del self._tracks[tid]
        for i in unmatched:
            self._tracks[self._nxt] = {
                "box": boxes[i], "cls": cls_ids[i], "lost": 0
            }
            self._nxt += 1
        return [
            (tid, t["box"], t["cls"])
            for tid, t in self._tracks.items()
            if t["lost"] == 0
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Motion filter
# ─────────────────────────────────────────────────────────────────────────────

class MotionFilter:
    def __init__(self, min_disp: float = 1.0, history: int = 25):
        self.min_disp = min_disp
        self.history  = history
        self._hy: Dict[int, deque] = {}

    def update(self, tid: int, y: float):
        if tid not in self._hy:
            self._hy[tid] = deque(maxlen=self.history)
        self._hy[tid].append(y)

    def has_history(self, tid: int) -> bool:
        h = self._hy.get(tid)
        return h is not None and len(h) >= max(5, self.history // 4)

    def is_moving(self, tid: int) -> bool:
        h = self._hy.get(tid)
        if not h or len(h) < max(5, self.history // 4):
            return False
        return abs(float(h[-1]) - float(h[0])) >= self.min_disp

    def evict(self, tid: int):
        self._hy.pop(tid, None)


# ─────────────────────────────────────────────────────────────────────────────
# Perspective transformer
# ─────────────────────────────────────────────────────────────────────────────

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        self.m = cv2.getPerspectiveTransform(
            source.astype(np.float32), target.astype(np.float32)
        )

    def transform_points(self, pts: np.ndarray) -> np.ndarray:
        if len(pts) == 0:
            return pts
        p = pts.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(p, self.m).reshape(-1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Named-pipe writer  (feeds annotated frames to live_streaming.cpp)
# ─────────────────────────────────────────────────────────────────────────────

class PipeWriter:
    """
    Creates a POSIX named pipe (FIFO) and writes annotated BGR frames into it.

    Frame wire format (for FFmpeg rawvideo demuxer on the C++ side):
        [4 bytes LE uint32: width][4 bytes LE uint32: height][width*height*3 bytes BGR]

    The C++ live_streaming program must use this as its RTSP_CAMERA_URL:
        pipe:<path>   →  read via  ffmpeg -f rawvideo -pix_fmt bgr24
                                         -video_size <W>x<H> -framerate <FPS>
                                         -i <path>  …

    Note:  open() on a FIFO blocks until a reader connects.
    That is intentional — start live_streaming first (or concurrently),
    and this writer will proceed once the C++ process opens the read end.
    """

    def __init__(self, path: str = PIPE_PATH):
        self.path = path
        self._fd: Optional[int] = None
        self._log = logging.getLogger("PipeWriter")

    def open(self):
        """Create the FIFO (if missing) and open it for writing."""
        if not os.path.exists(self.path):
            os.mkfifo(self.path, mode=0o666)
            self._log.info(f"Created FIFO: {self.path}")
        else:
            if not stat.S_ISFIFO(os.stat(self.path).st_mode):
                raise RuntimeError(
                    f"{self.path} exists but is not a FIFO. "
                    "Remove it and restart."
                )
            self._log.info(f"Using existing FIFO: {self.path}")

        self._log.info(
            f"Waiting for live_streaming to open the read end of {self.path} …"
        )
        # O_WRONLY | O_NONBLOCK lets us open without a reader present on Linux.
        # We then switch back to blocking for normal writes.
        import fcntl
        self._fd = os.open(self.path, os.O_WRONLY | os.O_NONBLOCK)
        flags = fcntl.fcntl(self._fd, fcntl.F_GETFL)
        fcntl.fcntl(self._fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
        self._log.info("Pipe open — streaming annotated frames to live_streaming")

    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Write one BGR frame into the pipe.
        Returns False if the pipe is broken (live_streaming exited).
        """
        if self._fd is None:
            return False
        h, w = frame.shape[:2]
        header = struct.pack(PIPE_FRAME_HEADER_FMT, w, h)
        payload = frame.tobytes()
        try:
            os.write(self._fd, header + payload)
            return True
        except BrokenPipeError:
            self._log.warning("Pipe broken — live_streaming may have exited")
            return False
        except OSError as exc:
            self._log.error(f"Pipe write error: {exc}")
            return False

    def close(self):
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        # Clean up the FIFO file
        try:
            os.unlink(self.path)
        except OSError:
            pass
        self._log.info(f"Pipe closed and removed: {self.path}")


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class LocalSpeedEstimator:
    """
    SpeedEstimator for local CPU use with RTSP pipe output.

    Annotated frames are written into a named pipe so live_streaming.cpp
    can forward them as rtsp://<server>:8554/mycamera via MediaMTX.
    """

    def __init__(
        self,
        camera_id: str,
        db,
        source: str,
        speed_limit: int = 80,
        yolo_model_path: str = YOLO_MODEL_PATH,
        show_preview: bool = True,
        save_output: bool = False,
        pipe_path: str = PIPE_PATH,
    ):
        self.camera_id    = camera_id
        self.db           = db
        self.source       = source
        self.speed_limit  = speed_limit
        self.show_preview = show_preview
        self.save_output  = save_output
        self._log = logging.getLogger(f"LocalSpeedEstimator[{camera_id}]")

        # ── Named-pipe writer → live_streaming.cpp → RTSP ────────────────
        self._pipe_writer = PipeWriter(pipe_path) if pipe_path else None

        # ── Load YOLO on CPU ──────────────────────────────────────────────
        self._log.info(f"Loading YOLO model from: {yolo_model_path}")
        self.model = YOLO(yolo_model_path, task="detect")

        # .to("cpu") only works for PyTorch (.pt) models.
        # OpenVINO / ONNX / TensorRT exported models are format-native and
        # do NOT support .to() — calling it raises TypeError.
        # Detect format once here; store the right device string for predict().
        _path_lower = str(yolo_model_path).lower()
        _is_pytorch = _path_lower.endswith(".pt")
        if _is_pytorch:
            self.model.to("cpu")
            self._infer_device = "cpu"   # passed into every model() call
        else:
            # OpenVINO / ONNX etc. — device is handled internally by the runtime.
            # Passing device="" lets OpenVINO pick CPU automatically.
            self._infer_device = None
        self._log.info(
            "YOLO model loaded  "
            f"(format: {'pytorch' if _is_pytorch else 'exported/openvino'}  "
            f"device: {self._infer_device or 'model-default/CPU'})"
        )

        # ── Load calibration ──────────────────────────────────────────────
        calibration = fetch_calibration(db, camera_id)
        if not calibration:
            raise ValueError(
                f"No calibration document for camera '{camera_id}'. "
                f"Check the '{CALIBRATION_COL}' collection."
            )

        src = np.array(calibration["source_points"], dtype=np.float32)
        self._cal_w_m = float(calibration["real_width_m"])
        self._cal_l_m = float(calibration["real_length_m"])
        self._cal_fw  = int(calibration.get("frame_width", 0))
        self._cal_fh  = int(calibration.get("frame_height", 0))
        # Per-camera blur threshold (falls back to global default)
        self._blur_thr = float(calibration.get("blur_threshold", BLUR_THRESHOLD))

        tgt = np.array([
            [0, 0],
            [self._cal_w_m, 0],
            [self._cal_w_m, self._cal_l_m],
            [0, self._cal_l_m],
        ], dtype=np.float32)

        self.source_pts  = src
        self.transformer = ViewTransformer(src, tgt)

        # ── Per-frame state ───────────────────────────────────────────────
        self.fps            = 25.0
        self.coord_history: Dict[int, deque] = {}
        self.motion_filter: Optional[MotionFilter] = None
        self.tracker        = None

        # DB throttle
        self._last_insert_ts: Dict[int, float] = {}
        # Stale-track eviction (defaultdict — no KeyError risk)
        self._frames_since_seen: Dict[int, int] = defaultdict(int)

    # ── public entry point ────────────────────────────────────────────────

    def run(self):
        """Open the source, process every frame, show preview, write DB."""
        # ── open capture ─────────────────────────────────────────────────
        src = self.source
        # Allow "0", "1" etc. as webcam indices
        if src.isdigit():
            src = int(src)

        self._log.info(f"Opening source: {src}")
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            # RTSP sources: try again with explicit backend
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            self._log.error(f"Cannot open source: {self.source}")
            return

        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        real_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._log.info(f"Stream: {real_w}x{real_h} @ {self.fps:.1f} fps")

        # ── open named pipe → live_streaming.cpp ─────────────────────────
        if self._pipe_writer is not None:
            try:
                self._pipe_writer.open()
                self._log.info(
                    f"Pipe ready: {self._pipe_writer.path}  "
                    f"({real_w}x{real_h} bgr24 @ {self.fps:.1f} fps)"
                )
            except Exception as exc:
                self._log.error(f"Cannot open pipe: {exc} — pipe output disabled")
                self._pipe_writer = None

        # ── rescale calibration if frame size differs ─────────────────────
        if (
            self._cal_fw > 0 and self._cal_fh > 0
            and (real_w != self._cal_fw or real_h != self._cal_fh)
        ):
            sx = real_w / self._cal_fw
            sy = real_h / self._cal_fh
            self._log.info(f"Rescaling calibration points: sx={sx:.4f} sy={sy:.4f}")
            self.source_pts[:, 0] *= sx
            self.source_pts[:, 1] *= sy
            tgt = np.array([
                [0, 0],
                [self._cal_w_m, 0],
                [self._cal_w_m, self._cal_l_m],
                [0, self._cal_l_m],
            ], dtype=np.float32)
            self.transformer = ViewTransformer(self.source_pts, tgt)

        window = max(int(self.fps), 5)
        self.coord_history = defaultdict(lambda: deque(maxlen=window))
        self.motion_filter = MotionFilter(MIN_SPEED_MPS, window)

        if HAS_SV:
            self.tracker = sv.ByteTrack(frame_rate=int(self.fps))
        else:
            self.tracker = SimpleTracker()

        # ── optional video writer ─────────────────────────────────────────
        writer = None
        if self.save_output:
            out_path = f"output_{self.camera_id}_{datetime.now():%Y%m%d_%H%M%S}.mp4"
            fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
            writer   = cv2.VideoWriter(out_path, fourcc, self.fps, (real_w, real_h))
            self._log.info(f"Saving annotated output to: {out_path}")

        # ── preview window ────────────────────────────────────────────────
        win_name = f"Speed Cam: {self.camera_id}"
        if self.show_preview:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            prev_w = min(real_w, PREVIEW_MAX_W)
            prev_h = int(prev_w / real_w * real_h)
            cv2.resizeWindow(win_name, prev_w, prev_h)
            self._log.info(f"Preview window: '{win_name}'  (press Q or Esc to quit)")

        frame_idx     = 0
        show_preview  = self.show_preview   # local copy — can be toggled off

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    self._log.info("End of stream / no more frames")
                    break

                annotated = None
                if frame_idx % FRAME_STEP == 0:
                    annotated = self._process_frame(frame, frame_idx)

                display = annotated if annotated is not None else frame

                # ── write to file ─────────────────────────────────────────
                if writer is not None and annotated is not None:
                    writer.write(annotated)

                # ── write to named pipe → live_streaming.cpp → RTSP ──────
                if self._pipe_writer is not None:
                    ok = self._pipe_writer.write_frame(display)
                    if not ok:
                        self._log.warning("Pipe broken — stopping pipe output")
                        self._pipe_writer.close()
                        self._pipe_writer = None

                # ── preview ───────────────────────────────────────────────
                if show_preview:
                    prev = display
                    if real_w > PREVIEW_MAX_W:
                        scale = PREVIEW_MAX_W / real_w
                        prev  = cv2.resize(
                            display,
                            (PREVIEW_MAX_W, int(real_h * scale)),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    cv2.imshow(win_name, prev)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        self._log.info("Quit key pressed — stopping")
                        break

                frame_idx += 1

        except KeyboardInterrupt:
            self._log.info("Interrupted by user (Ctrl+C)")
        finally:
            cap.release()
            if writer:
                writer.release()
            if self._pipe_writer:
                self._pipe_writer.close()
            if self.show_preview:
                try:
                    cv2.destroyWindow(win_name)
                except Exception:
                    pass
            self._log.info(f"Done. Processed {frame_idx} frames.")

    # ── _process_frame (identical logic to main_k.py SpeedEstimator) ──────

    def _process_frame(
        self, frame: np.ndarray, frame_idx: int
    ) -> Optional[np.ndarray]:
        """
        Runs YOLO detection → tracking → speed estimation → DB insert.
        Returns the annotated frame, or None if the frame was skipped (blurry).
        """
        if is_blurry(frame, self._blur_thr):
            return None

        # ── YOLO inference (CPU) ──────────────────────────────────────────
        try:
            # Build kwargs — only pass device= for PyTorch models.
            # OpenVINO/ONNX models ignore or reject an explicit device arg.
            _infer_kwargs = dict(
                conf=CONF_THRESHOLD,
                iou=NMS_IOU,
                verbose=False,   # suppress per-frame stdout spam on CPU
            )
            if self._infer_device is not None:
                _infer_kwargs["device"] = self._infer_device
            results = self.model(frame, **_infer_kwargs)[0]
        except Exception:
            self._log.error("YOLO inference failed", exc_info=True)
            return None

        tracks      = self._get_tracks(results)
        active_tids = {int(tid) for tid, _, _ in tracks}
        ts          = datetime.utcnow()
        now_mono    = time.monotonic()
        tracks_render = []

        self._evict_stale_tracks(active_tids)

        for tid, box, cid in tracks:
            tid = int(tid)
            try:
                cls_name        = VEHICLE_CLASSES.get(cid, "vehicle")
                x1, y1, x2, y2 = box
                bc              = np.array([[(x1 + x2) / 2, y2]])
                real_y          = float(
                    self.transformer.transform_points(bc)[0][1]
                )

                self.motion_filter.update(tid, real_y)
                self.coord_history[tid].append(real_y)

                if not self.motion_filter.has_history(tid):
                    continue
                if not self.motion_filter.is_moving(tid):
                    continue

                hist = self.coord_history[tid]
                if len(hist) < max(5, int(self.fps / 2)):
                    continue

                # Correct speed formula: distance / elapsed_seconds
                elapsed_s = len(hist) / self.fps
                dist      = abs(float(hist[-1]) - float(hist[0]))
                speed_mps = dist / elapsed_s
                if speed_mps < MIN_SPEED_MPS:
                    continue
                speed_kmh    = round(speed_mps * 3.6, 1)
                overspeeding = speed_kmh > self.speed_limit

                # Throttled DB insert
                last_ts = self._last_insert_ts.get(tid, 0.0)
                if (now_mono - last_ts) >= DB_INSERT_COOLDOWN_S:
                    record = {
                        "camera_id"   : self.camera_id,
                        "tracker_id"  : tid,
                        "vehicle_type": cls_name,
                        "speed_kmh"   : speed_kmh,
                        "speed_limit" : self.speed_limit,
                        "overspeeding": overspeeding,
                        "bbox"        : [round(v, 1) for v in box],
                        "frame_index" : frame_idx,
                        "timestamp"   : ts,
                        "created_at"  : datetime.utcnow(),
                        "source"      : "local_cpu",
                    }
                    insert_speed_estimate(self.db, record)
                    self._last_insert_ts[tid] = now_mono

                tracks_render.append({
                    "tid": tid, "box": box,
                    "cls": cls_name, "speed": speed_kmh,
                })

                if overspeeding:
                    self._log.warning(
                        f"OVERSPEED tid={tid} {cls_name} "
                        f"{speed_kmh} km/h (limit {self.speed_limit})"
                    )
                else:
                    self._log.info(
                        f"Vehicle tid={tid} {cls_name} {speed_kmh} km/h"
                    )

            except Exception:
                self._log.error(f"Track tid={tid} error", exc_info=True)

        return self._draw_frame(frame, tracks_render)

    # ── annotation ────────────────────────────────────────────────────────

    def _draw_frame(
        self, frame: np.ndarray, render_info: list
    ) -> np.ndarray:
        canvas = frame.copy()
        h, w   = canvas.shape[:2]
        fs     = max(0.5, w / 1280 * 0.65)
        th     = max(1, int(w / 1280 * 2))

        # Draw calibration polygon
        if self.source_pts is not None:
            try:
                poly = self.source_pts.astype(np.int32)
                cv2.polylines(canvas, [poly], True, (0, 220, 255), th + 1, cv2.LINE_AA)
                cv2.line(canvas, tuple(poly[0]), tuple(poly[1]),
                         (60, 255, 100), th + 1, cv2.LINE_AA)
                cv2.line(canvas, tuple(poly[3]), tuple(poly[2]),
                         (60, 80, 255),  th + 1, cv2.LINE_AA)
                cv2.putText(canvas, "ENTRY",
                            (int(poly[0][0]) + 8, int(poly[0][1]) - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, (60, 255, 100), th, cv2.LINE_AA)
                cv2.putText(canvas, "EXIT",
                            (int(poly[3][0]) + 8, int(poly[3][1]) - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, (60, 80, 255),  th, cv2.LINE_AA)
            except Exception:
                pass

        # Draw tracked vehicles
        for info in render_info:
            try:
                x1, y1, x2, y2 = map(int, info["box"])
                speed_kmh = info.get("speed")
                cls_name  = info.get("cls", "vehicle")
                tid       = info["tid"]
                overspeed = speed_kmh is not None and speed_kmh > self.speed_limit
                color     = (0, 60, 255) if overspeed else (0, 255, 128)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, th)
                label = (
                    f"#{tid} {cls_name}  {speed_kmh} km/h"
                    if speed_kmh is not None
                    else f"#{tid} {cls_name}"
                )
                if overspeed:
                    label += " !"
                (tw, lh), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, fs, th
                )
                lx = x1
                ly = max(y1 - lh - 8, lh + 4)
                cv2.rectangle(canvas, (lx, ly - lh - 4), (lx + tw + 8, ly + 4),
                              color, -1)
                cv2.putText(canvas, label, (lx + 4, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th, cv2.LINE_AA)
            except Exception:
                pass

        # HUD overlay
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (380, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
        cv2.putText(canvas, f"CAM: {self.camera_id}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    fs * 0.8, (255, 255, 255), th, cv2.LINE_AA)
        cv2.putText(canvas, f"Limit: {self.speed_limit} km/h",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    fs * 0.8, (255, 255, 255), th, cv2.LINE_AA)
        cv2.putText(canvas, f"Tracked: {len(render_info)}",
                    (10, 74), cv2.FONT_HERSHEY_SIMPLEX,
                    fs * 0.8, (255, 255, 255), th, cv2.LINE_AA)
        cv2.putText(canvas, "MODE: CPU LOCAL",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    fs * 0.7, (0, 200, 255), th, cv2.LINE_AA)
        return canvas

    # ── tracker abstraction ───────────────────────────────────────────────

    def _get_tracks(self, results) -> list:
        try:
            if HAS_SV:
                dets = sv.Detections.from_ultralytics(results)
                if len(dets) > 0:
                    mask = np.array(
                        [int(c) in VEHICLE_CLASSES for c in dets.class_id]
                    )
                    dets = dets[mask]
                dets = self.tracker.update_with_detections(dets)
                if len(dets) == 0:
                    return []
                return list(zip(
                    dets.tracker_id.tolist(),
                    dets.xyxy.tolist(),
                    [int(c) for c in dets.class_id],
                ))
            else:
                boxes, cls_ids = [], []
                for box in results.boxes:
                    cid = int(box.cls[0])
                    if cid in VEHICLE_CLASSES:
                        boxes.append(box.xyxy[0].tolist())
                        cls_ids.append(cid)
                return self.tracker.update(boxes, cls_ids)
        except Exception:
            self._log.error("Tracking error", exc_info=True)
            return []

    # ── stale-track eviction ──────────────────────────────────────────────

    def _evict_stale_tracks(self, active_tids: Set[int]):
        all_known = set(self.coord_history.keys())
        for tid in all_known:
            if tid not in active_tids:
                self._frames_since_seen[tid] += 1
                if self._frames_since_seen[tid] >= STALE_TRACK_FRAMES:
                    del self.coord_history[tid]
                    self.motion_filter.evict(tid)
                    self._last_insert_ts.pop(tid, None)
                    del self._frames_since_seen[tid]
            else:
                self._frames_since_seen.pop(tid, None)


# ─────────────────────────────────────────────────────────────────────────────
# ▼▼▼  RUN CONFIG — edit these values, then just run: python speed_estimator_1.py
# ─────────────────────────────────────────────────────────────────────────────

# Source: RTSP URL, local video file path, or webcam index as a string ("0")
RUN_SOURCE      = "rtsp://admin:admin@192.168.1.2:554/rtsp/streaming?channel=01&subtype=0"   # ← change this

# camera_id must match the entry in calibration.json (or MongoDB as fallback)
RUN_CAMERA_ID   = "CAM_001"                          # ← change this

# Local calibration file — loaded first, MongoDB used only if this is missing.
RUN_CALIBRATION_FILE = "calibration.json"            # ← change path if needed

# Speed limit displayed on the overlay and used for overspeed flagging (km/h)
RUN_SPEED_LIMIT = 80

# YOLO model path.  Use "yolov8n.pt" if you don't have the OpenVINO model.
RUN_MODEL       = YOLO_MODEL_PATH

# Show cv2.imshow preview window? Set False for headless / SSH / pipe mode.
RUN_SHOW_PREVIEW = False

# Save the annotated frames to output_<camera_id>_<timestamp>.mp4?
RUN_SAVE_OUTPUT  = False

# Process every Nth frame.  1 = every frame (best accuracy).
# Raise to 2 or 3 on a slow laptop to keep up in real-time.
RUN_FRAME_STEP   = 1

# ── Pipe output → live_streaming.cpp → RTSP ─────────────────────────────────
# Named pipe (FIFO) path that this script writes annotated BGR frames into.
# live_streaming.cpp reads from this path as its "camera" input and forwards
# the video to MediaMTX as rtsp://<EC2_SERVER>:8554/mycamera.
#
# Set to None (or "") to disable pipe output entirely (standalone preview mode).
RUN_PIPE_PATH   = "/tmp/speed_pipe"

# ─────────────────────────────────────────────────────────────────────────────


def main():
    global FRAME_STEP
    FRAME_STEP = RUN_FRAME_STEP
    if FRAME_STEP > 1:
        logger.info(
            f"Frame step: {FRAME_STEP} "
            f"(processing every {FRAME_STEP}th frame — CPU speed mode)"
        )

    # Apply calibration file path from RUN CONFIG
    global LOCAL_CALIBRATION_FILE
    LOCAL_CALIBRATION_FILE = RUN_CALIBRATION_FILE

    logger.info("Connecting to MongoDB…")
    mongo_client, db = make_db()

    try:
        estimator = LocalSpeedEstimator(
            camera_id       = RUN_CAMERA_ID,
            db              = db,
            source          = RUN_SOURCE,
            speed_limit     = RUN_SPEED_LIMIT,
            yolo_model_path = RUN_MODEL,
            show_preview    = RUN_SHOW_PREVIEW,
            save_output     = RUN_SAVE_OUTPUT,
            pipe_path       = RUN_PIPE_PATH or "",
        )
    except ValueError as exc:
        logger.error(str(exc))
        mongo_client.close()
        sys.exit(1)

    try:
        estimator.run()
    finally:
        mongo_client.close()
        logger.info("MongoDB connection closed")


if __name__ == "__main__":
    main()