# app.py
# Lightweight, improved detection server (no heavy ML).
# - POST frames to /detect as multipart form field 'image' (jpeg)
# - GET /detect returns status
# - UI: templates/login.html and templates/index.html (existing)
#
# Requirements: flask, opencv-python, numpy, python-dotenv

import os
import traceback
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ---------------- CONFIG ----------------
SECRET_KEY = os.environ.get("SECRET_KEY", "dev_secret")
DEMO_USER = os.environ.get("DEMO_USERNAME", "demo")
DEMO_PASS = os.environ.get("DEMO_PASSWORD", "password")

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", 5000))

# image size used by client (canvas/video). Keep in sync with templates/index.html
INPUT_W = int(os.environ.get("INPUT_W", 640))
INPUT_H = int(os.environ.get("INPUT_H", 480))

# Background subtractor / morphology / detection thresholds
MOG_HISTORY = int(os.environ.get("MOG_HISTORY", 300))
MOG_VARTH = float(os.environ.get("MOG_VARTH", 20.0))
DETECT_SHADOWS = os.environ.get("DETECT_SHADOWS", "1") != "0"

KERNEL_SIZE = int(os.environ.get("KERNEL_SIZE", 7))
MIN_AREA = int(os.environ.get("MIN_AREA", 1200))   # increase to reduce wall false positives
MAX_AREA = int(os.environ.get("MAX_AREA", 500000))

CONF_BASE = float(os.environ.get("CONF_BASE", 0.30))  # baseline confidence
IOU_NMS_THRESH = float(os.environ.get("IOU_NMS_THRESH", 0.45))

# Temporal persistence
RECENT_BUFFER_MAX = int(os.environ.get("RECENT_BUFFER_MAX", 6))
PERSISTENCE_REQ = int(os.environ.get("PERSISTENCE_REQ", 2))

# Tracker params
MAX_DISAPPEARED = int(os.environ.get("MAX_DISAPPEARED", 10))

# ---------------- APP ----------------
app = Flask(__name__, template_folder="templates")
app.secret_key = SECRET_KEY

# ---------------- Globals ----------------
# Background subtractor persists across requests so detection is stable across frames
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=MOG_HISTORY, varThreshold=MOG_VARTH, detectShadows=DETECT_SHADOWS)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))

# Centroid tracker to stabilize IDs
class CentroidTracker:
    def __init__(self, max_disappeared=MAX_DISAPPEARED):
        self.next_object_id = 0
        self.objects = dict()       # id -> centroid (x,y)
        self.disappeared = dict()   # id -> frames disappeared
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        if object_id in self.objects: del self.objects[object_id]
        if object_id in self.disappeared: del self.disappeared[object_id]

    def update(self, rects):
        """
        rects: list of boxes [x1,y1,x2,y2]
        returns dict mapping object_id -> box (or None)
        """
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return {oid: None for oid in self.objects.keys()}

        input_centroids = []
        for (x1, y1, x2, y2) in rects:
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids.append((cX, cY))

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
            # map all registered ids to rects
            start_id = self.next_object_id - len(input_centroids)
            return {oid: rects[i] for i, oid in enumerate(range(start_id, self.next_object_id))}

        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid] for oid in object_ids]

        D = np.zeros((len(object_centroids), len(input_centroids)), dtype=np.float32)
        for i, oc in enumerate(object_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = np.linalg.norm(np.array(oc) - np.array(ic))

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        assigned_rows = set()
        assigned_cols = set()
        mapping = dict()

        for r, c in zip(rows, cols):
            if r in assigned_rows or c in assigned_cols:
                continue
            oid = object_ids[r]
            self.objects[oid] = input_centroids[c]
            self.disappeared[oid] = 0
            mapping[oid] = rects[c]
            assigned_rows.add(r); assigned_cols.add(c)

        for r in range(0, D.shape[0]):
            if r not in assigned_rows:
                oid = object_ids[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

        for c in range(0, len(input_centroids)):
            if c not in assigned_cols:
                self.register(input_centroids[c])
                nid = self.next_object_id - 1
                mapping[nid] = rects[c]

        return mapping

tracker = CentroidTracker(max_disappeared=MAX_DISAPPEARED)

# HOG people detector (lightweight fallback for human detection)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Recent buffer for temporal persistence
RECENT_BUFFER = []  # list of dicts {frame_gray:, boxes: [...]}

# ---------------- Utility functions ----------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    inter = interW * interH
    boxAArea = (boxA[2]-boxA[0]+1) * (boxA[3]-boxA[1]+1)
    boxBArea = (boxB[2]-boxB[0]+1) * (boxB[3]-boxB[1]+1)
    union = float(boxAArea + boxBArea - inter)
    if union <= 0: return 0.0
    return inter / union

def nms(boxes, scores, iou_thresh=IOU_NMS_THRESH):
    if len(boxes) == 0:
        return []
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        rest = idxs[1:]
        removes = []
        for j_index, j in enumerate(rest):
            if iou(boxes[i], boxes[j]) > iou_thresh:
                removes.append(j_index)
        idxs = np.delete(idxs, [0] + [r+1 for r in removes])
    return keep

def boxes_iou_inter(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    inter = interW * interH
    areaA = max(1, (boxA[2]-boxA[0])*(boxA[3]-boxA[1]))
    areaB = max(1, (boxB[2]-boxB[0])*(boxB[3]-boxB[1]))
    return inter / float(min(areaA, areaB))

def filter_by_shape(cnt, min_area=MIN_AREA, max_area=MAX_AREA):
    area = cv2.contourArea(cnt)
    if area < min_area or area > max_area:
        return False
    x,y,w,h = cv2.boundingRect(cnt)
    rect_area = w*h
    if rect_area <= 0:
        return False
    solidity = area / float(rect_area)
    aspect = float(w)/float(h) if h>0 else 10.0
    if solidity < 0.25:
        return False
    if aspect > 6.0 or aspect < 0.15:
        return False
    return True

def compute_flow_motion(prev_gray, cur_gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None,
                                        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                                        poly_n=5, poly_sigma=1.2, flags=0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    motion_mask = (mag > 1.2).astype('uint8') * 255
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_DILATE, kernel, iterations=1)
    return motion_mask

# ---------------- Improved precise detector (drop-in) ----------------
def improved_detect_from_bgr_precise(img_bgr):
    """
    Precise lightweight detector:
      - combine BG mask, frame-diff, optical flow
      - shape filtering (solidity, aspect)
      - temporal persistence across RECENT_BUFFER
      - HOG fallback for humans
      - NMS + tracker
    """
    global RECENT_BUFFER
    try:
        h, w = img_bgr.shape[:2]
        work = img_bgr.copy()
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

        # 1) BG subtractor mask
        fg = bg_subtractor.apply(work)
        if DETECT_SHADOWS:
            _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 2) frame-diff
        prev_frame = None
        if len(RECENT_BUFFER) > 0 and isinstance(RECENT_BUFFER[-1], dict) and RECENT_BUFFER[-1].get("frame_gray") is not None:
            prev_frame = RECENT_BUFFER[-1]["frame_gray"]
        frame_diff_mask = np.zeros_like(fg)
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, gray)
            _, frame_diff_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            frame_diff_mask = cv2.morphologyEx(frame_diff_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # 3) optical flow mask
        motion_mask = np.zeros_like(fg)
        if prev_frame is not None:
            motion_mask = compute_flow_motion(prev_frame, gray)

        # 4) combined: (fg & frame_diff) OR (fg & motion)
        combined1 = cv2.bitwise_and(fg, frame_diff_mask)
        combined2 = cv2.bitwise_and(fg, motion_mask)
        combined = cv2.bitwise_or(combined1, combined2)
        combined = cv2.medianBlur(combined, 5)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 5) contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_boxes = []
        candidate_scores = []
        for cnt in contours:
            if not filter_by_shape(cnt):
                continue
            x,y,wc,hc = cv2.boundingRect(cnt)
            if wc < 10 or hc < 10:
                continue
            pad = 4
            x1 = max(0, x-pad); y1 = max(0, y-pad)
            x2 = min(img_bgr.shape[1]-1, x + wc + pad); y2 = min(img_bgr.shape[0]-1, y + hc + pad)
            candidate_boxes.append([x1,y1,x2,y2])
            mask_roi = combined[y1:y2, x1:x2]
            density = (np.count_nonzero(mask_roi) / float(max(1, mask_roi.size)))
            score = float(min(0.99, CONF_BASE + density))
            candidate_scores.append(score)

        # 6) HOG fallback if nothing detected
        if len(candidate_boxes) == 0:
            scale = 1.0
            if max(w, h) > 800:
                scale = 800.0 / max(w, h)
            small = cv2.resize(img_bgr, (int(w*scale), int(h*scale)))
            rects, weights = hog.detectMultiScale(small, winStride=(8,8), padding=(8,8), scale=1.05)
            for (rx, ry, rw, rh), wt in zip(rects, weights):
                x1 = int(rx/scale); y1 = int(ry/scale)
                x2 = int((rx+rw)/scale); y2 = int((ry+rh)/scale)
                candidate_boxes.append([x1,y1,x2,y2])
                # weight can be an array/float
                wv = float(wt[0]) if hasattr(wt, "__len__") else float(wt)
                candidate_scores.append(float(min(0.99, 0.3 + wv)))

        # 7) NMS
        keep = nms(candidate_boxes, candidate_scores, iou_thresh=IOU_NMS_THRESH)
        final_boxes = [candidate_boxes[i] for i in keep]
        final_scores = [candidate_scores[i] for i in keep]

        # 8) temporal buffer
        RECENT_BUFFER.append({"frame_gray": gray, "boxes": final_boxes})
        if len(RECENT_BUFFER) > RECENT_BUFFER_MAX:
            RECENT_BUFFER.pop(0)

        # presence persistence
        persistent = []
        for i, box in enumerate(final_boxes):
            count = 0
            for entry in RECENT_BUFFER:
                for b in entry["boxes"]:
                    if boxes_iou_inter(b, box) > 0.4:
                        count += 1
                        break
            if count >= PERSISTENCE_REQ:
                persistent.append((box, final_scores[i]))
            else:
                if final_scores[i] > 0.85:
                    persistent.append((box, final_scores[i]))

        # 9) tracker update and build detections
        mapping = tracker.update([b for b,_ in persistent])
        dets = []
        for oid, box in mapping.items():
            if box is None:
                continue
            score = CONF_BASE
            for b,s in persistent:
                if b == box:
                    score = s; break
            dets.append({
                "track_id": int(oid),
                "class_id": 0,
                "class_name": "object",
                "confidence": float(score),
                "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
            })
        return dets
    except Exception as e:
        print("precise_detect error:", e)
        return []

# ---------------- Routes ----------------
@app.route("/", methods=["GET"])
def route_login():
    return render_template("login.html")

@app.route("/auth", methods=["POST"])
def route_auth():
    user = request.form.get("username")
    pw = request.form.get("password")
    if user == DEMO_USER and pw == DEMO_PASS:
        session["user"] = user
        return redirect(url_for("route_home"))
    return render_template("login.html", error="Invalid credentials")

@app.route("/home", methods=["GET"])
def route_home():
    if "user" not in session:
        return redirect(url_for("route_login"))
    return render_template("index.html", user=session.get("user"))

@app.route("/logout")
def route_logout():
    session.pop("user", None)
    return redirect(url_for("route_login"))

@app.route("/detect", methods=["GET", "POST"])
def route_detect():
    # GET -> status
    if request.method == "GET":
        return jsonify({"status":"ready", "note":"POST 'image' as multipart/form-data (jpeg)"})

    # POST -> accept multipart form field 'image'
    try:
        if "image" not in request.files:
            return jsonify({"error":"no 'image' file provided"}), 400
        data = request.files["image"].read()
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error":"could not decode image"}), 400

        detections = improved_detect_from_bgr_precise(img)
        return jsonify({"detections": detections})
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"error": str(e), "trace": tb}), 500

# ---------------- Run ----------------
if __name__ == "__main__":
    # run without reloader to avoid one-drive / reloader issues
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)
