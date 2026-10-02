#!/usr/bin/env python3
# dev/creator=tubakhxn
import argparse
import math
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit("Missing packages. Run:  pip install opencv-python mediapipe numpy")

HERE = Path(__file__).resolve().parent
HAND_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task")

BODY = (225, 112, 128)
DARK = (122, 52, 72)
EDGE = (62, 22, 34)
HAIR = (96, 34, 56)
ORANGE = (40, 140, 255)
CYAN = (255, 225, 70)
GOLD = (60, 195, 255)
PINK = (190, 120, 255)
WHITE = (240, 240, 240)
DIM = (150, 160, 170)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def clip(x, a, b):
    return max(a, min(b, x))


def lerp(a, b, k):
    return a + (b - a) * k


def light(c):
    return tuple(int(min(255, v * 1.22 + 28)) for v in c)


def smooth_damp(cur, tgt, vel, st, dt):
    st = max(1e-4, st)
    w = 2.0 / st
    x = w * dt
    e = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x ** 3)
    ch = cur - tgt
    tmp = (vel + w * ch) * dt
    vel = (vel - w * tmp) * e
    return tgt + (ch + tmp) * e, vel


def find_model():
    for base in (HERE / "models", Path.cwd() / "models", HERE, Path.cwd()):
        p = base / "hand_landmarker.task"
        if p.is_file() and p.stat().st_size > 1_000_000:
            return p
    p = HERE / "models" / "hand_landmarker.task"
    p.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading hand model (first run only)...")
    tmp = p.with_suffix(".part")
    req = urllib.request.Request(HAND_URL, headers={"User-Agent": "gesture-mech"})
    with urllib.request.urlopen(req, timeout=40) as r, open(tmp, "wb") as f:
        f.write(r.read())
    tmp.replace(p)
    return p


class HandTracker:
    def __init__(self):
        import mediapipe as mp
        self.mp = mp
        self.kind = None
        self.last_ts = -1
        try:
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision
            opts = vision.HandLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(find_model())),
                running_mode=vision.RunningMode.VIDEO, num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5)
            self.lm = vision.HandLandmarker.create_from_options(opts)
            self.kind = "tasks"
        except Exception as e:
            print("Tasks API failed (%s) -> trying legacy mediapipe hands" % e)
            self.hands = mp.solutions.hands.Hands(
                max_num_hands=1, min_detection_confidence=0.5,
                min_tracking_confidence=0.5)
            self.kind = "legacy"

    def detect(self, rgb, ts_ms):
        rgb = np.ascontiguousarray(rgb)
        if self.kind == "tasks":
            ts_ms = max(ts_ms, self.last_ts + 1)
            self.last_ts = ts_ms
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            res = self.lm.detect_for_video(img, ts_ms)
            if res.hand_landmarks:
                return np.array([[p.x, p.y] for p in res.hand_landmarks[0]])
            return None
        res = self.hands.process(rgb)
        if res.multi_hand_landmarks:
            return np.array([[p.x, p.y] for p in res.multi_hand_landmarks[0].landmark])
        return None


TIPS = (4, 8, 12, 16, 20)


def hand_features(pts):
    def d(a, b):
        return float(np.linalg.norm(pts[a] - pts[b])) + 1e-6
    o = [clip((d(4, 17) / d(5, 17) - 0.9) / 0.9, 0, 1)]
    for tip, mcp in ((8, 5), (12, 9), (16, 13), (20, 17)):
        o.append(clip((d(tip, 0) / d(mcp, 0) - 1.05) / 0.75, 0, 1))
    pinch = d(4, 8) / d(0, 9)
    v = pts[9] - pts[0]
    roll = math.degrees(math.atan2(v[0], -v[1]))
    return o, pinch, roll, pts.mean(axis=0)


def classify(o, pinch):
    th, ix, md, rg, pk = o
    if pinch < 0.30 and md > 0.5:
        return "COMPILING"
    if th > 0.6 and pk > 0.65 and ix < 0.4 and md < 0.4 and rg < 0.4:
        return "AUTO MODE"
    if all(v < 0.4 for v in (ix, md, rg, pk)):
        return "PUSH-UPS"
    if all(v > 0.6 for v in (ix, md, rg, pk)):
        return "SCANNING CODEBASE"
    if ix > 0.6 and md > 0.6 and rg > 0.6 and pk < 0.4:
        return "POWER STANCE"
    if ix > 0.6 and md < 0.45 and rg < 0.45 and pk < 0.45:
        return "RUNNING IMPLEMENTATION"
    if ix > 0.6 and md > 0.6 and rg < 0.45 and pk < 0.45:
        return "DEPLOYING BUILD"
    if ix > 0.6 and pk > 0.6 and md < 0.45 and rg < 0.45:
        return "MIGRATING MEMORY"
    return None


def pose_target(st, o, lean, t, ph):
    th, ix, md, rg, pk = o
    s = math.sin
    P = dict(al=10 + 150 * ix, ar=10 + 150 * md, el=15 + 70 * th, er=15 + 70 * th,
             tl=5 + 16 * rg, tr=5 + 16 * pk, kl=6 + 62 * rg, kr=6 + 62 * pk,
             head=(th - 0.5) * 12, lean=lean * 0.5, eye=0.7 + 0.3 * ix, bob=0.0)
    if st == "IDLE":
        P.update(al=12, ar=12, el=20, er=20, tl=5, tr=5, kl=6, kr=6,
                 lean=1.5 * s(t * 0.6), head=4 * s(t * 0.7),
                 bob=0.003 * s(t * 1.6), eye=0.55)
    elif st == "RUNNING IMPLEMENTATION":
        w = s(ph)
        c = math.cos(ph)
        P.update(al=40 + 40 * w, ar=40 - 40 * w, el=82, er=82, tl=8, tr=8,
                 kl=8 + 58 * max(0, w) ** 1.2, kr=8 + 58 * max(0, -w) ** 1.2,
                 bob=abs(c) * 0.012, lean=10, head=-3 * w, eye=1.0)
    elif st == "POWER STANCE":  # standing tall, hands on hips
        P.update(al=40, ar=40, el=-72, er=-72, tl=9, tr=9, kl=6, kr=6,
                 lean=0, head=5 * s(t * 0.8), bob=0.004 * s(t * 1.8), eye=1.0)
    elif st == "SCANNING CODEBASE":
        P.update(al=55 + 5 * s(t * 1.8), ar=55 - 5 * s(t * 1.8), el=40, er=40,
                 head=18 * s(t * 1.4), tl=8, tr=8, kl=6, kr=6, eye=1.0)
    elif st == "DEPLOYING BUILD":
        P.update(ar=95, er=8, al=25, el=105, head=-5, tl=8, tr=8, kl=8, kr=8, eye=1.0)
    elif st == "COMPILING":
        P.update(al=70, ar=70, el=95, er=95, head=0, tl=6, tr=6, kl=8, kr=8,
                 eye=0.6 + 0.4 * (0.5 + 0.5 * s(t * 6)))
    elif st == "MIGRATING MEMORY":
        w = s(t * 3.5)
        P.update(al=100 + 28 * w, ar=100 - 28 * w, el=30, er=30, head=6 * w,
                 tl=10, tr=10, kl=10, kr=10, eye=1.0)
    return P


def dirv(side, ang):
    a = math.radians(ang)
    return (side * math.sin(a), math.cos(a))


def rot(p, ang, c=(0.0, 0.0)):
    a = math.radians(ang)
    ca, sa = math.cos(a), math.sin(a)
    x, y = p[0] - c[0], p[1] - c[1]
    return (c[0] + x * ca - y * sa, c[1] + x * sa + y * ca)


def limb(img, p0, p1, w0, w1, col):
    p0 = np.array(p0, float)
    p1 = np.array(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d) + 1e-6
    n = np.array([-d[1], d[0]]) / L
    if n[0] + n[1] > 0:
        n = -n

    def poly(k, off=0.0):
        a = p0 + n * (w0 * k + off)
        b = p1 + n * (w1 * k + off)
        c = p1 - n * (w1 * k - off)
        e = p0 - n * (w0 * k - off)
        return np.array([a, b, c, e], np.int32)
    cv2.fillPoly(img, [poly(1.2)], EDGE, cv2.LINE_AA)
    cv2.fillPoly(img, [poly(1.0)], col, cv2.LINE_AA)
    cv2.fillPoly(img, [poly(0.28, w0 * 0.45)], light(col), cv2.LINE_AA)


def joint(img, p, r, col=BODY):
    p = (int(p[0]), int(p[1]))
    cv2.circle(img, p, int(r * 1.15), EDGE, -1, cv2.LINE_AA)
    cv2.circle(img, p, int(r), col, -1, cv2.LINE_AA)
    cv2.circle(img, (p[0] - int(r * .3), p[1] - int(r * .3)), max(1, int(r * .38)),
               light(col), -1, cv2.LINE_AA)


def plate(img, pts, col, edge=EDGE, th=2):
    a = np.array(pts, np.int32)
    cv2.fillPoly(img, [a], col, cv2.LINE_AA)
    cv2.polylines(img, [a], True, edge, th, cv2.LINE_AA)


POSE_KEYS = dict(al=12, ar=12, el=18, er=18, tl=5, tr=5, kl=6, kr=6,
                 head=0, lean=0, eye=0.6, bob=0)


class Robot:
    def __init__(self):
        self.p = dict(POSE_KEYS)
        self.v = {k: 0.0 for k in POSE_KEYS}
        self.ox = None
        self.ox_v = 0.0
        self.phase = 0.0
        self.pu_phase = 0.0
        self.mode = "IDLE"
        self.pts = {}

    def update(self, target, ox, dt):
        for key, tv in target.items():
            st = 0.07 if key in ("al", "ar", "tl", "tr", "kl", "kr", "bob") else 0.14
            self.p[key], self.v[key] = smooth_damp(self.p[key], tv, self.v[key], st, dt)
        if self.ox is None:
            self.ox = ox
        else:
            self.ox, self.ox_v = smooth_damp(self.ox, ox, self.ox_v, 0.25, dt)

    def draw(self, rl, gl, W, H, t):
        if self.mode == "PUSH-UPS":
            return self.draw_pushup(rl, gl, W, H, t)
        s = 0.62 * H
        P = self.p
        floor = 0.90 * H
        leg = {}
        for sd, tk, kk in ((-1, "tl", "kl"), (1, "tr", "kr")):
            hip = (sd * 0.055, 0.05)
            kn = tuple(np.add(hip, np.multiply(dirv(sd, P[tk]), 0.20)))
            an = tuple(np.add(kn, np.multiply(dirv(sd, P[tk] - P[kk]), 0.20)))
            leg[sd] = (hip, kn, an)
        ymax = max(leg[-1][2][1], leg[1][2][1]) + 0.035
        py = floor - ymax * s - P["bob"] * s
        ox = self.ox

        def S(p):
            return (ox + p[0] * s, py + p[1] * s)

        lean = P["lean"]
        hl = lean + P["head"] * 0.5
        sh = {sd: rot((sd * 0.14, -0.29), lean) for sd in (-1, 1)}
        neck = rot((0, -0.31), lean)
        hc = rot((0, -0.395), hl, neck)
        e = clip(P["eye"], 0, 1)

        c = (int(ox), int(floor + 0.012 * H))
        ax = (int(0.2 * H), int(0.035 * H))
        cv2.ellipse(gl, c, ax, 0, 0, 360, (90, 70, 20), 1, cv2.LINE_AA)
        a0 = (t * 50) % 360
        cv2.ellipse(gl, c, ax, 0, a0, a0 + 70, CYAN, 2, cv2.LINE_AA)
        cv2.ellipse(gl, c, ax, 0, a0 + 180, a0 + 250, ORANGE, 2, cv2.LINE_AA)

        for sd in (-1, 1):
            hip, kn, an = leg[sd]
            limb(rl, S(hip), S(kn), 0.038 * s, 0.031 * s, BODY)
            joint(rl, S(kn), 0.036 * s, DARK)
            limb(rl, S(kn), S(an), 0.031 * s, 0.025 * s, BODY)
            f0 = S((an[0] - 0.04, an[1] + 0.012))
            f1 = S((an[0] + 0.04, an[1] + 0.033))
            plate(rl, [(f0[0], f0[1]), (f1[0], f0[1]), (f1[0] + 4, f1[1]),
                       (f0[0] - 4, f1[1])], DARK)
            cv2.line(rl, (int(f0[0]), int(f1[1] - 3)), (int(f1[0]), int(f1[1] - 3)),
                     ORANGE, 2, cv2.LINE_AA)
        plate(rl, [S((-0.10, -0.02)), S((0.10, -0.02)), S((0.145, 0.08)),
                   S((-0.145, 0.08))], DARK)
        a = S((-0.13, 0.07))
        b = S((0.13, 0.07))
        cv2.line(rl, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), ORANGE, 2, cv2.LINE_AA)
        T_ = [rot(p, lean) for p in ((-0.15, -0.32), (0.15, -0.32), (0.065, -0.12),
                                     (0.095, -0.02), (-0.095, -0.02), (-0.065, -0.12))]
        plate(rl, [S(p) for p in T_], BODY)
        ch = [rot(p, lean) for p in ((-0.105, -0.29), (0.105, -0.29), (0.06, -0.14),
                                     (-0.06, -0.14))]
        plate(rl, [S(p) for p in ch], DARK)
        core = S(rot((0, -0.21), lean))
        cv2.circle(rl, (int(core[0]), int(core[1])), int(0.034 * s), EDGE, -1, cv2.LINE_AA)
        cv2.circle(rl, (int(core[0]), int(core[1])), int(0.026 * s),
                   tuple(int(v * (0.4 + 0.6 * e)) for v in CYAN), -1, cv2.LINE_AA)
        cv2.circle(gl, (int(core[0]), int(core[1])), int(0.03 * s),
                   tuple(int(v * 0.8 * e) for v in CYAN), -1, cv2.LINE_AA)
        for i in range(3):
            a = S(rot((-0.04 + i * 0.03, -0.085), lean))
            b = S(rot((-0.025 + i * 0.03, -0.065), lean))
            cv2.rectangle(rl, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), ORANGE, -1)

        hands = {}
        for sd, ak, ek in ((-1, "al", "el"), (1, "ar", "er")):
            sp = sh[sd]
            el = (sp[0] + dirv(sd, P[ak])[0] * 0.165, sp[1] + dirv(sd, P[ak])[1] * 0.165)
            hd = (el[0] + dirv(sd, P[ak] + P[ek])[0] * 0.155,
                  el[1] + dirv(sd, P[ak] + P[ek])[1] * 0.155)
            limb(rl, S(sp), S(el), 0.032 * s, 0.027 * s, BODY)
            joint(rl, S(el), 0.030 * s, DARK)
            limb(rl, S(el), S(hd), 0.030 * s, 0.023 * s, BODY)
            joint(rl, S(hd), 0.035 * s, DARK)
            joint(rl, S(sp), 0.050 * s, BODY)
            cv2.circle(rl, (int(S(sp)[0]), int(S(sp)[1])), int(0.026 * s), ORANGE, 2,
                       cv2.LINE_AA)
            hands[sd] = S(hd)

        def HP(pts):
            return [S(rot(p, hl, neck)) for p in pts]

        limb(rl, S(neck), S((neck[0], neck[1] - 0.03)), 0.026 * s, 0.026 * s, DARK)
        plate(rl, HP([(-0.088, -0.47), (0.088, -0.47), (0.092, -0.385),
                      (0.075, -0.36), (-0.075, -0.36), (-0.092, -0.385)]), HAIR)
        plate(rl, HP([(-0.068, -0.45), (0.068, -0.45), (0.07, -0.37),
                      (0.04, -0.34), (-0.04, -0.34), (-0.07, -0.37)]), BODY)
        vz = HP([(-0.055, -0.41), (0.055, -0.41), (0.055, -0.38), (-0.055, -0.38)])
        vcol = tuple(int(v * (0.35 + 0.65 * e)) for v in CYAN)
        plate(rl, vz, vcol)
        gc = tuple(int(v * 0.7 * e) for v in CYAN)
        plate(gl, vz, gc, gc)
        for sd in (-1, 1):
            q = S(rot((sd * 0.045, -0.365), hl, neck))
            cv2.circle(rl, (int(q[0]), int(q[1])), max(2, int(0.011 * s)), PINK, -1, cv2.LINE_AA)
        plate(rl, HP([(-0.09, -0.40), (-0.092, -0.47), (-0.05, -0.50), (0.05, -0.50),
                      (0.092, -0.47), (0.09, -0.40), (0.075, -0.43), (0.03, -0.425),
                      (-0.01, -0.445), (-0.045, -0.43), (-0.07, -0.425)]), HAIR)
        hi = HP([(-0.04, -0.485), (0.03, -0.49), (0.045, -0.475), (-0.02, -0.47)])
        cv2.fillPoly(rl, [np.array(hi, np.int32)], light(HAIR), cv2.LINE_AA)
        for sd in (-1, 1):
            q = S(rot((sd * 0.083, -0.395), hl, neck))
            cv2.circle(rl, (int(q[0]), int(q[1])), max(3, int(0.016 * s)), EDGE, -1, cv2.LINE_AA)
            cv2.circle(rl, (int(q[0]), int(q[1])), max(2, int(0.011 * s)), ORANGE, -1, cv2.LINE_AA)

        self.pts = dict(head=S(hc), hand_l=hands[-1], hand_r=hands[1],
                        foot_l=S(leg[-1][2]), foot_r=S(leg[1][2]),
                        top=S(rot((0, -0.5), lean)), pelvis=S((0, 0)), floor=floor)

    # ---------------------------- push-ups (side view) ----------------------
    def draw_pushup(self, rl, gl, W, H, t):
        s = 0.62 * H
        floor = 0.90 * H
        u = 0.5 + 0.5 * math.cos(self.pu_phase)          # 1 = arms straight
        hs = 0.09 + 0.22 * u                              # shoulder height
        Lb = 0.60
        hx = clip(max(self.ox, 0.52 * H), 0, W - 0.25 * H)
        e = 1.0

        def P(v):
            return (hx + v[0] * s, floor + v[1] * s)

        Sv = np.array([-0.02, -hs])
        Hv = np.array([0.0, -0.02])
        dx = math.sqrt(max(0.01, Lb * Lb - (hs - 0.03) ** 2))
        Av = np.array([-0.02 - dx, -0.03])
        Hip = Sv + (Av - Sv) * 0.52 + np.array([0, -0.012])
        fwd = (Sv - Av) / (np.linalg.norm(Sv - Av) + 1e-6)

        def elbow(Sx, Hx):
            d = np.linalg.norm(Hx - Sx) + 1e-6
            dv = (Hx - Sx) / d
            mid = (Sx + Hx) / 2
            hh = math.sqrt(max(0.0, 0.16 ** 2 - (d / 2) ** 2))
            pp = np.array([-dv[1], dv[0]])
            if pp[0] > 0:
                pp = -pp
            return mid + pp * hh

        # pad
        c = (int(hx - 0.30 * s), int(floor + 0.012 * H))
        ax = (int(0.46 * s), int(0.03 * H))
        cv2.ellipse(gl, c, ax, 0, 0, 360, (90, 70, 20), 1, cv2.LINE_AA)
        a0 = (t * 50) % 360
        cv2.ellipse(gl, c, ax, 0, a0, a0 + 70, CYAN, 2, cv2.LINE_AA)
        cv2.ellipse(gl, c, ax, 0, a0 + 180, a0 + 250, ORANGE, 2, cv2.LINE_AA)

        # far arm (darker, slightly behind)
        off = np.array([-0.035, 0.0])
        Ef = elbow(Sv + off, Hv + off)
        limb(rl, P(Sv + off), P(Ef), 0.028 * s, 0.024 * s, DARK)
        limb(rl, P(Ef), P(Hv + off), 0.026 * s, 0.021 * s, DARK)
        # legs + shoes
        limb(rl, P(Hip), P(Av), 0.042 * s, 0.03 * s, BODY)
        joint(rl, P((Hip + Av) / 2), 0.034 * s, DARK)
        sh0 = P(Av + np.array([-0.045, -0.02]))
        sh1 = P(Av + np.array([0.02, 0.03]))
        plate(rl, [(sh0[0], sh0[1]), (sh1[0], sh0[1]), (sh1[0], sh1[1]), (sh0[0], sh1[1])], DARK)
        cv2.line(rl, (int(sh0[0]), int(sh1[1] - 3)), (int(sh1[0]), int(sh1[1] - 3)),
                 ORANGE, 2, cv2.LINE_AA)
        # pelvis + torso
        limb(rl, P(Sv + (Av - Sv) * 0.40), P(Sv + (Av - Sv) * 0.62), 0.072 * s, 0.07 * s, DARK)
        limb(rl, P(Sv), P(Sv + (Av - Sv) * 0.42), 0.066 * s, 0.062 * s, BODY)
        cp = P(Sv + (Av - Sv) * 0.2 + np.array([0, -0.03]))
        cv2.circle(rl, (int(cp[0]), int(cp[1])), int(0.02 * s), CYAN, -1, cv2.LINE_AA)
        cv2.circle(gl, (int(cp[0]), int(cp[1])), int(0.026 * s), (120, 100, 30), -1, cv2.LINE_AA)
        # near arm
        En = elbow(Sv, Hv)
        limb(rl, P(Sv), P(En), 0.031 * s, 0.026 * s, BODY)
        joint(rl, P(En), 0.029 * s, DARK)
        limb(rl, P(En), P(Hv), 0.029 * s, 0.022 * s, BODY)
        hp = P(Hv)
        plate(rl, [(hp[0] - 0.035 * s, hp[1] + 0.02 * s), (hp[0] + 0.05 * s, hp[1] + 0.02 * s),
                   (hp[0] + 0.05 * s, hp[1] - 0.01 * s), (hp[0] - 0.035 * s, hp[1] - 0.01 * s)], DARK)
        joint(rl, P(Sv), 0.05 * s, BODY)
        cv2.circle(rl, (int(P(Sv)[0]), int(P(Sv)[1])), int(0.026 * s), ORANGE, 2, cv2.LINE_AA)
        # head (facing right) with short hair
        hc = P(Sv + fwd * 0.115 + np.array([0.0, -0.035]))
        cx, cy = int(hc[0]), int(hc[1])
        r = int(0.07 * s)
        cv2.circle(rl, (cx, cy), int(r * 1.12), EDGE, -1, cv2.LINE_AA)
        cv2.circle(rl, (cx, cy), r, BODY, -1, cv2.LINE_AA)
        cv2.ellipse(rl, (cx, cy), (int(r * 1.1), int(r * 1.1)), 0, 150, 390, HAIR, -1, cv2.LINE_AA)
        cv2.ellipse(rl, (cx - int(r * 0.45), cy + int(r * 0.15)), (int(r * 0.6), int(r * 0.85)),
                    0, 90, 270, HAIR, -1, cv2.LINE_AA)
        plate(rl, [(cx - int(r * 0.2), cy - int(r * 1.05)), (cx + int(r * 1.0), cy - int(r * 0.2)),
                   (cx + int(r * 0.55), cy - int(r * 0.2)), (cx + int(r * 0.2), cy - int(r * 0.5))],
              HAIR, HAIR, 1)
        v0 = (cx + int(r * 0.2), cy - int(r * 0.12))
        v1 = (cx + int(r * 0.98), cy + int(r * 0.25))
        cv2.rectangle(rl, v0, v1, CYAN, -1)
        cv2.rectangle(gl, v0, v1, (140, 120, 35), -1)
        cv2.circle(rl, (cx - int(r * 0.15), cy + int(r * 0.2)), max(3, int(r * 0.22)), ORANGE, -1, cv2.LINE_AA)
        if u < 0.25:
            for k in range(3):
                a = t * 9 + k * 2.1
                cv2.circle(gl, (cx - int(r * 1.4 + 6 * math.sin(a)), cy - int(r * (1.2 + 0.4 * k))),
                           2, GOLD, -1, cv2.LINE_AA)

        self.pts = dict(head=(cx, cy), hand_l=P(Hv + off), hand_r=P(Hv),
                        foot_l=P(Av), foot_r=P(Av + np.array([0.02, 0])),
                        top=(cx, cy), pelvis=P(Hip), floor=floor)


def draw_strings(gl, tips, robot, t, fingers_open):
    targets = [robot.pts["head"], robot.pts["hand_l"], robot.pts["hand_r"],
               robot.pts["foot_l"], robot.pts["foot_r"]]
    for i, (tp, tg) in enumerate(zip(tips, targets)):
        p0 = np.array(tp, float)
        p2 = np.array(tg, float)
        dist = np.linalg.norm(p2 - p0)
        mid = (p0 + p2) / 2
        for k in range(4):
            off = (k - 1.5) * 7
            sag = dist * (0.10 + 0.05 * k) + 3 * math.sin(t * 2 + i + k)
            p1 = mid + np.array([off * 0.3, sag + off])
            ts = np.linspace(0, 1, 28)[:, None]
            curve = (1 - ts) ** 2 * p0 + 2 * (1 - ts) * ts * p1 + ts ** 2 * p2
            br = 0.55 + 0.45 * fingers_open[i]
            cv2.polylines(gl, [curve.astype(np.int32)], False,
                          tuple(int(v * br) for v in GOLD), 1, cv2.LINE_AA)
        cv2.circle(gl, (int(p2[0]), int(p2[1])), 4, GOLD, -1, cv2.LINE_AA)
    for a, b in zip(tips[:-1], tips[1:]):
        a = np.array(a, float)
        b = np.array(b, float)
        m = (a + b) / 2 + np.array([0, 0.25 * np.linalg.norm(a - b)])
        ts = np.linspace(0, 1, 14)[:, None]
        cv = (1 - ts) ** 2 * a + 2 * (1 - ts) * ts * m + ts ** 2 * b
        cv2.polylines(gl, [cv.astype(np.int32)], False, (30, 110, 150), 1, cv2.LINE_AA)


def T(img, s, x, y, sc=0.45, col=WHITE, th=1):
    cv2.putText(img, s, (int(x), int(y)), FONT, sc, col, th, cv2.LINE_AA)


def tint(img, x0, y0, x1, y1, k=0.35):
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(img.shape[1], int(x1)), min(img.shape[0], int(y1))
    if x1 > x0 and y1 > y0:
        r = img[y0:y1, x0:x1]
        cv2.convertScaleAbs(r, dst=r, alpha=k)


def notch_box(img, x0, y0, x1, y1, col, cut=10, th=1):
    pts = np.array([(x0 + cut, y0), (x1 - cut, y0), (x1, y0 + cut), (x1, y1 - cut),
                    (x1 - cut, y1), (x0 + cut, y1), (x0, y1 - cut), (x0, y0 + cut)], np.int32)
    cv2.polylines(img, [pts], True, col, th, cv2.LINE_AA)


TASKS = ["string_solver", "gesture_compiler", "puppet_daemon", "mesh_optimizer",
         "agent_router", "tensor_cache", "diff_engine", "prompt_linter"]
IDEAS = ["puppets that dream when idle", "robots that review PRs", "strings as syntax",
         "agents that pair with agents", "hands as a compiler", "gestures as git"]
JOBS = ["migrate memory store (96GB)", "compact heap (1.8G)",
        "train epoch 12/40 (512 batches)", "rebuild embedding index", "sync weights (4.2G)"]
VERB = {
    "RUNNING IMPLEMENTATION": ("run",), "PUSH-UPS": ("lift",),
    "SCANNING CODEBASE": ("run", "idea"), "DEPLOYING BUILD": ("run", "lift"),
    "COMPILING": ("run",), "MIGRATING MEMORY": ("lift",),
    "POWER STANCE": ("idea",), "MANUAL CONTROL": ("run", "idea"), "IDLE": ("idea",),
}


class Terminal:
    def __init__(self):
        self.lines = [("run", "running string_solver•••"), ("done", "done · 0.4s")]
        self.next = 0.0

    def tick(self, t, status):
        if t < self.next:
            return
        self.next = t + random.uniform(0.5, 1.1)
        if self.lines[-1][0] in ("run", "lift") and random.random() < 0.6:
            if self.lines[-1][0] == "lift":
                self.lines.append(("done", "done · %.1fGB in %.1fs" % (random.uniform(.5, 96), random.uniform(.5, 5))))
            else:
                self.lines.append(("done", "done · %d batches in %.1fs" % (random.choice((64, 128, 512)), random.uniform(.5, 5))))
        else:
            k = random.choice(VERB.get(status, ("run",)))
            if k == "run":
                self.lines.append(("run", "running %s%s" % (random.choice(TASKS), "•" * random.randint(3, 7))))
            elif k == "lift":
                self.lines.append(("lift", "lifting: " + random.choice(JOBS)))
            else:
                self.lines.append(("idea", "IDEA %d: %s" % (random.randint(1, 40), random.choice(IDEAS))))
        self.lines = self.lines[-9:]

    def draw(self, img, sc):
        W = img.shape[1]
        w, lh = int(350 * sc), int(19 * sc)
        x0, y0 = W - w - int(18 * sc), int(18 * sc)
        h = lh * 9 + int(14 * sc)
        tint(img, x0, y0, x0 + w, y0 + h, 0.3)
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (205, 210, 214), 1, cv2.LINE_AA)
        for i, (k, s) in enumerate(self.lines[-9:]):
            y = y0 + int(lh * (i + 1))
            T(img, "$", x0 + 8, y, 0.4 * sc, ORANGE)
            col = (150, 235, 150) if k == "done" else (235, 238, 238)
            T(img, s[:46], x0 + 24 * sc, y, 0.4 * sc, col)


def draw_hud(img, rob, label, fps, fo, hand_on, auto_on, sc, t):
    H, W = img.shape[:2]
    ox = rob.ox
    head = rob.pts["head"]
    bw, bh = int(0.62 * H), int(0.095 * H)
    bx0 = int(clip(ox - 0.30 * H, 10, W - bw - 10))
    by0 = int(max(14, head[1] - 0.20 * H))
    tint(img, bx0, by0, bx0 + bw, by0 + bh, 0.45)
    notch_box(img, bx0, by0, bx0 + bw, by0 + bh, (235, 235, 235), int(10 * sc), 2)
    T(img, "AGENT #0380075", bx0 + 16 * sc, by0 + 0.38 * bh, 0.62 * sc, ORANGE, 2)
    T(img, "STATUS: " + label, bx0 + 16 * sc, by0 + 0.78 * bh, 0.55 * sc, WHITE, 2)
    if rob.mode == "PUSH-UPS":
        T(img, "REPS %02d" % int(rob.pu_phase / (2 * math.pi)), bx0 + bw - 110 * sc,
          by0 + 0.38 * bh, 0.6 * sc, CYAN, 2)
    rx = int(clip(ox - 0.34 * H, 10, W))
    for y in range(int(0.12 * H), int(0.92 * H), 9):
        cv2.line(img, (rx, y), (rx, y + 2), (150, 150, 150), 1)
    for y in range(int(0.12 * H), int(0.92 * H), int(0.08 * H)):
        cv2.line(img, (rx, y), (rx + 8, y), (200, 200, 200), 1)
    fy = int(0.955 * H)
    for i, s in enumerate(("PWR %03d%%" % (80 + 10 * math.sin(t)),
                           "LINK %s" % ("OK" if hand_on else "--"),
                           "MODE %s" % ("AUTO" if auto_on else "MANUAL"))):
        T(img, s, ox - 0.30 * H + i * 0.2 * H, fy, 0.38 * sc, DIM)
    m, L = int(14 * sc), int(40 * sc)
    for (cx, cy, dx, dy) in ((m, m, 1, 1), (W - m, m, -1, 1), (m, H - m, 1, -1), (W - m, H - m, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * L, cy), CYAN, 2, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), CYAN, 2, cv2.LINE_AA)
    T(img, "ROBO // GESTURE LINK", 30 * sc, 40 * sc, 0.6 * sc, CYAN, 2)
    T(img, "HAND-CONTROLLED PUPPET MECH", 30 * sc, 62 * sc, 0.4 * sc, DIM)
    T(img, "FPS %02d  Q EXIT  F FULL  H HAND  A AUTO" % fps, 30 * sc, 84 * sc, 0.38 * sc, DIM)
    names = ("THUMB", "INDEX", "MIDDLE", "RING", "PINKY")
    bx = W - int(235 * sc)
    by = H - int(150 * sc)
    tint(img, bx - 12, by - 26 * sc, W - 14 * sc, H - 14 * sc, 0.4)
    T(img, "FINGER CHANNELS", bx, by - 8 * sc, 0.4 * sc, CYAN)
    for i, n in enumerate(names):
        y = by + i * 20 * sc + 10 * sc
        T(img, n, bx, y + 4, 0.35 * sc, DIM)
        cv2.rectangle(img, (int(bx + 64 * sc), int(y - 5)), (int(bx + 204 * sc), int(y + 5)), (90, 90, 90), 1)
        cv2.rectangle(img, (int(bx + 64 * sc), int(y - 5)),
                      (int(bx + (64 + 140 * fo[i]) * sc), int(y + 5)), ORANGE, -1)
    if not hand_on:
        s = "SHOW YOUR HAND TO TAKE CONTROL"
        (tw, _), _ = cv2.getTextSize(s, FONT, 0.7 * sc, 2)
        T(img, s, (W - tw) / 2, H * 0.1, 0.7 * sc, ORANGE if int(t * 2) % 2 else WHITE, 2)


AUTO_LIST = ["RUNNING IMPLEMENTATION", "RUNNING IMPLEMENTATION", "PUSH-UPS", "PUSH-UPS",
             "POWER STANCE", "SCANNING CODEBASE", "DEPLOYING BUILD",
             "MIGRATING MEMORY", "COMPILING"]


class Auto:
    def __init__(self):
        self.state = "POWER STANCE"
        self.until = 0.0

    def step(self, t):
        if t >= self.until:
            nxt = random.choice([x for x in AUTO_LIST if x != self.state])
            self.state = nxt
            d = {"PUSH-UPS": (6, 9), "RUNNING IMPLEMENTATION": (4, 7)}.get(nxt, (3, 5))
            self.until = t + random.uniform(*d)
        return self.state


DEMO_SEQ = [("MANUAL CONTROL", [.6, .5, .5, .5, .5]), ("SCANNING CODEBASE", [1, 1, 1, 1, 1]),
            ("PUSH-UPS", [0, 0, 0, 0, 0]), ("RUNNING IMPLEMENTATION", [0, 1, 0, 0, 0]),
            ("DEPLOYING BUILD", [0, 1, 1, 0, 0]), ("MIGRATING MEMORY", [0, 1, 0, 0, 1])]


def fake_hand(o, cx, cy, roll, size, W, H):
    pts = np.zeros((21, 2))
    pts[5], pts[9], pts[13], pts[17] = (-.42, -.95), (0, -1), (.4, -.95), (.76, -.82)
    pts[1], pts[2] = (-.38, -.18), (-.62, -.45)
    ang = math.radians(lerp(-60, -150, o[0]))
    dv = np.array([math.cos(ang), math.sin(ang)])
    pts[3] = pts[2] + .38 * dv
    pts[4] = pts[3] + .32 * dv
    for base, mcp, a0 in ((5, 5, -100), (9, 9, -90), (13, 13, -82), (17, 17, -72)):
        i = (5, 9, 13, 17).index(base)
        c = (1 - o[i + 1]) * 85
        p = pts[mcp].copy()
        for k, (ln, f) in enumerate(((.45, .3), (.28, .8), (.24, 1.3))):
            a = math.radians(a0 + c * f)
            p = p + ln * np.array([math.cos(a), math.sin(a)])
            pts[mcp + 1 + k] = p
    r = math.radians(roll)
    R = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    pts = (pts @ R.T) * size + np.array([cx, cy])
    return pts / np.array([W, H])


def build_overlays(W, H):
    scan = np.zeros((H, W, 3), np.uint8)
    scan[::3] = 14
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    vig = np.clip(1.15 - 0.45 * d ** 2, 0.4, 1.0)
    vig = np.dstack([vig] * 3)
    return scan, (vig * 255).astype(np.uint8)


# ============================ main ===========================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--snap", help="demo only: save a PNG and exit")
    ap.add_argument("--snap-t", type=float, default=4.0, help=argparse.SUPPRESS)
    a = ap.parse_args()

    cap = tracker = None
    if not a.demo:
        try:
            tracker = HandTracker()
        except ImportError:
            sys.exit("Missing mediapipe. Run:  pip install mediapipe")
        cap = cv2.VideoCapture(a.cam, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(a.cam)
        if not cap.isOpened():
            cap = cv2.VideoCapture(a.cam)
        if not cap.isOpened():
            sys.exit("Could not open camera %d. Try --cam 1 and check permissions." % a.cam)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    W = a.width
    H = int(W * 9 / 16)
    sc = H / 720.0
    scan, vig = build_overlays(W, H)
    robot, term, auto = Robot(), Terminal(), Auto()
    status, cand, cnt = "MANUAL CONTROL", None, 0
    smooth = None
    fo_s = [0.0] * 5
    key_auto = False
    prev_mode = None
    fps, t_prev, t0 = 0.0, time.time(), time.time()
    show_hand, full = True, False
    win = "GESTURE MECH"
    if not a.snap:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, W, H)
    n = 0
    while True:
        now = time.time()
        dt = clip(now - t_prev, 1e-3, 0.05)
        t_prev = now
        t = now - t0
        if a.demo:
            bg = np.full((H, W, 3), (26, 22, 20), np.uint8)
            cv2.circle(bg, (int(W * .62), int(H * .75)), int(H * .32), (60, 62, 66), -1)
            frame = cv2.GaussianBlur(bg, (0, 0), 25)
            if a.snap:
                t = a.snap_t
            name, o = DEMO_SEQ[int(t / 3.0) % len(DEMO_SEQ)]
            cx = W * (0.62 + 0.08 * math.sin(t))
            lm = fake_hand(o, cx, H * 0.55, 8 * math.sin(t * 1.3), H * 0.2, W, H)
        else:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped giving frames.")
                break
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
            small = cv2.resize(frame, (640, int(640 * H / W)))
            lm = tracker.detect(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), int(t * 1000))
        frame = cv2.convertScaleAbs(frame, alpha=0.62, beta=0)

        hand_on = lm is not None
        fo = [0.0] * 5
        tips = None
        roll = 0.0
        ctr = None
        if hand_on:
            px = lm * np.array([W, H])
            if smooth is None:
                smooth = px
            else:
                dist = float(np.linalg.norm(px - smooth, axis=1).mean())
                k = clip(0.25 + dist / (0.04 * H), 0.25, 0.9)
                smooth = lerp(smooth, px, k)
            o, pinch, roll, ctr = hand_features(smooth)
            fo_s = [lerp(a_, b_, 0.45) for a_, b_ in zip(fo_s, o)]
            fo = fo_s
            g = classify(fo, pinch)
            if g == cand:
                cnt += 1
            else:
                cand, cnt = g, 1
            if cnt >= 4:
                status = g if g else "MANUAL CONTROL"
            tips = [tuple(smooth[i]) for i in TIPS]
        else:
            smooth = None
            cand, cnt = None, 0
            status = "MANUAL CONTROL"

        auto_on = key_auto or (hand_on and status == "AUTO MODE") or (not hand_on)
        if auto_on:
            pose_status = auto.step(t)
            label = "AUTO / " + pose_status
            ox = W * (0.30 + 0.17 * math.sin(t * (0.9 if pose_status.startswith("RUN") else 0.3)))
            pose_o = [0.3] * 5
            lean = 0
        else:
            pose_status = status
            label = status
            ox = W * (0.17 + 0.30 * clip(ctr[0] / W, 0, 1))
            pose_o = fo
            lean = clip(roll, -35, 35)

        if pose_status != prev_mode:
            if pose_status == "PUSH-UPS":
                robot.pu_phase = 0.0
            prev_mode = pose_status
        robot.mode = pose_status
        if pose_status == "RUNNING IMPLEMENTATION":
            robot.phase += dt * 7.0
        elif pose_status == "PUSH-UPS":
            robot.pu_phase += dt * 2.4

        if pose_status == "PUSH-UPS":
            ox = clip(ox, 0.52 * H, W - 0.3 * H)
        target = pose_target(pose_status, pose_o, lean, t, robot.phase)
        robot.update(target, ox, dt)

        rl = np.zeros_like(frame)
        gl = np.zeros_like(frame)
        robot.draw(rl, gl, W, H, t)
        if hand_on:
            draw_strings(gl, tips, robot, t, fo)

        m = (rl.max(axis=2) > 0)[:, :, None]
        blend = cv2.addWeighted(frame, 0.15, rl, 0.85, 0)
        frame = np.where(m, blend, frame)
        src = cv2.addWeighted(rl, 0.45, gl, 1.0, 0)
        sm = cv2.resize(src, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        sm = cv2.GaussianBlur(sm, (0, 0), 5)
        frame = cv2.add(frame, cv2.resize(sm, (W, H), interpolation=cv2.INTER_LINEAR))
        frame = cv2.add(frame, gl)
        if hand_on and show_hand:
            for a_, b_ in ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                           (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                           (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)):
                cv2.line(frame, tuple(smooth[a_].astype(int)), tuple(smooth[b_].astype(int)),
                         (130, 130, 130), 1, cv2.LINE_AA)
            for i in TIPS:
                p = tuple(smooth[i].astype(int))
                cv2.circle(frame, p, 7, WHITE, -1, cv2.LINE_AA)
                cv2.circle(frame, p, 11, GOLD, 1, cv2.LINE_AA)

        term.tick(t, pose_status)
        term.draw(frame, sc)
        draw_hud(frame, robot, label, fps, fo, hand_on, auto_on, sc, t)
        frame = cv2.subtract(frame, scan)
        frame = cv2.multiply(frame, vig, scale=1 / 255.0)

        fps = lerp(fps, 1.0 / dt, 0.1)
        n += 1
        if a.snap:
            if n >= 12:
                cv2.imwrite(a.snap, frame)
                print("saved", a.snap, "status:", label)
                return
            continue
        cv2.imshow(win, frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("h"):
            show_hand = not show_hand
        if k == ord("a"):
            key_auto = not key_auto
        if k == ord("s"):
            fn = "mech_%d.png" % int(time.time())
            cv2.imwrite(fn, frame)
            print("saved", fn)
        if k == ord("f"):
            full = not full
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()