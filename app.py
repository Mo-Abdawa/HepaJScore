"""
HepaJScore v18 — Imaging-assisted liver assessment
==================================================
Trilingual (FR / EN / AR) medical decision-support prototype.

Preserves the proven v17 business logic:
  - APRI / FIB-4 computation
  - Focused peri-limbic jaundice index K
  - Optional OpenCV eye detection
  - Composite J-score + interpretation
  - PDF report, admin area, CSV export, rate limiting

Adds in v18:
  - Full FR/EN/AR internationalization with RTL
  - All fields optional except the photograph
  - Extra structured clinical fields (albumin, INR, viral load,
    fibrosis stage, cirrhosis, ultrasound, sex, notes)
  - PWA support (manifest + service worker + offline shell)
  - Modernized, premium UI (separate templates)

NOTE: research prototype — not a certified medical device.
"""

import os, io, csv, math, time, json, sqlite3, logging, datetime, random, string, base64
from logging.handlers import RotatingFileHandler
from collections import deque, defaultdict

from flask import (
    Flask, render_template, request, flash, Response, send_from_directory,
    jsonify, url_for, redirect, make_response
)
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from PIL import Image, ImageFilter, ImageOps

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Eye detection (MediaPipe + OpenCV fallback). Mandatory for J-score.
from eye_detection import detect_eyes, sclera_yellow_fraction, detector_available

# Keep OpenCV reference for legacy compute_jaundice_k_focused use
try:
    import cv2
except Exception:
    cv2 = None

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
VERSION         = "18.0"
CONSENT_REF     = os.environ.get("CONSENT_REF", "Ministère de la Santé MR – réf. 012-2025 (07/07/2025)")
CONSENT_VERSION = os.environ.get("CONSENT_VERSION", "v1.0-2025-07-07")
CONTACT_EMAIL   = os.environ.get("CONTACT_EMAIL", "contact@hepajscore.org")

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DATA_FOLDER   = os.path.join(BASE_DIR, "data")
DB_PATH       = os.path.join(DATA_FOLDER, "hepajscore.db")
I18N_PATH     = os.path.join(BASE_DIR, "translations", "i18n.json")

try:
    from PIL import features as PIL_features
    _HAS_WEBP = bool(PIL_features.check("webp"))
except Exception:
    _HAS_WEBP = False
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"} | ({"webp"} if _HAS_WEBP else set())
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------------------------------------------------------------
# i18n
# ----------------------------------------------------------------------
with open(I18N_PATH, encoding="utf-8") as f:
    TRANSLATIONS = json.load(f)
SUPPORTED_LANGS = list(TRANSLATIONS.keys())
DEFAULT_LANG    = "fr"


def get_lang():
    """Resolve UI language from query string, cookie, or Accept-Language."""
    lang = request.args.get("lang") or request.cookies.get("hs_lang")
    if lang in SUPPORTED_LANGS:
        return lang
    accept = (request.headers.get("Accept-Language") or "").lower()
    for code in SUPPORTED_LANGS:
        if accept.startswith(code):
            return code
    return DEFAULT_LANG


# ----------------------------------------------------------------------
# Rate limiting (naive, in-memory)
# ----------------------------------------------------------------------
RATE_LIMIT     = defaultdict(lambda: deque())
LIMIT_PER_MIN  = 10
LIMIT_PER_HOUR = 30


def is_rate_limited(ip: str) -> bool:
    now = time.time()
    dq = RATE_LIMIT[ip]
    while dq and now - dq[0] > 3600:
        dq.popleft()
    last_minute = [t for t in dq if now - t <= 60]
    if len(last_minute) >= LIMIT_PER_MIN or len(dq) >= LIMIT_PER_HOUR:
        return True
    dq.append(now)
    return False


# ----------------------------------------------------------------------
# Flask app
# ----------------------------------------------------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-prod")

ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("ADMIN_PASS", "admin")

if app.secret_key == "change-me-in-prod" or (ADMIN_USER, ADMIN_PASS) == ("admin", "admin"):
    print(
        "\n*** WARNING: HepaJScore is running with default SECRET_KEY / "
        "ADMIN_USER / ADMIN_PASS. ***\n"
        "*** Set these via environment variables before exposing this "
        "app beyond localhost. ***\n",
        flush=True,
    )

# Logging
log_path = os.path.join(DATA_FOLDER, "app.log")
handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s in %(module)s: %(message)s"))
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)


# ----------------------------------------------------------------------
# Database
# ----------------------------------------------------------------------
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_utc TEXT,
            lang TEXT,
            date_photo TEXT,
            date_biomarkers TEXT,
            patient_id TEXT,
            diag TEXT,
            sex TEXT,
            photo_filename TEXT,
            age REAL, alt REAL, ast REAL, plt REAL, afp REAL, ggt REAL, pa REAL,
            bt REAL, bd REAL, albumin REAL, inr REAL, viral_load REAL,
            uln_ast REAL,
            fibrosis_stage TEXT,
            cirrhosis TEXT,
            ultrasound TEXT,
            clinical_notes TEXT,
            jaundice_k REAL,
            apri REAL,
            fib4 REAL,
            j_score REAL,
            j_interpretation TEXT,
            patient_signature TEXT,
            consent INTEGER,
            consent_ref TEXT,
            consent_version TEXT,
            submission_ip TEXT,
            user_agent TEXT,
            model_version TEXT
        )
    """)
    con.commit()

    # Migration: add any missing columns (keeps old v17 DBs working)
    cur.execute("PRAGMA table_info(submissions)")
    cols = {r[1] for r in cur.fetchall()}
    new_cols = [
        ("lang", "TEXT"), ("sex", "TEXT"), ("albumin", "REAL"), ("inr", "REAL"),
        ("viral_load", "REAL"), ("fibrosis_stage", "TEXT"), ("cirrhosis", "TEXT"),
        ("ultrasound", "TEXT"), ("clinical_notes", "TEXT"),
        ("consent_ref", "TEXT"), ("consent_version", "TEXT"),
        ("submission_ip", "TEXT"), ("user_agent", "TEXT"),
    ]
    for name, typ in new_cols:
        if name not in cols:
            try:
                cur.execute(f"ALTER TABLE submissions ADD COLUMN {name} {typ}")
            except Exception:
                pass
    con.commit()
    con.close()


init_db()


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def to_float(v):
    if v is None or v == "":
        return None
    try:
        return float(str(v).replace(",", "."))
    except ValueError:
        return None


def to_date(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def generate_patient_id():
    return datetime.datetime.utcnow().strftime("P-%Y%m%d-") + "".join(
        random.choices(string.ascii_uppercase + string.digits, k=4))


def require_basic_auth(auth_header):
    if not auth_header or not auth_header.startswith("Basic "):
        return False
    try:
        user, pwd = base64.b64decode(
            auth_header.split(" ", 1)[1]).decode("utf-8").split(":", 1)
        return user == ADMIN_USER and pwd == ADMIN_PASS
    except Exception:
        return False


# ----------------------------------------------------------------------
# Scores & interpretation (preserved from v17)
# ----------------------------------------------------------------------
def compute_apri(ast, uln_ast, plt):
    if ast is None or plt in (None, 0):
        return None
    if not uln_ast:
        uln_ast = 40.0
    return round(((ast / uln_ast) * 100.0) / plt, 4)


def compute_fib4(age, ast, alt, plt):
    if None in (age, ast, alt, plt) or plt == 0 or alt <= 0:
        return None
    return round((age * ast) / (plt * math.sqrt(alt)), 4)


def classify_apri(apri):
    if apri is None:
        return None
    if apri < 0.5:
        return "normal"
    elif apri >= 2.0:
        return "cirrhosis"
    return "fibrosis"


def classify_fib4(fib4):
    if fib4 is None:
        return None
    if fib4 < 1.45:
        return "normal"
    elif fib4 >= 3.25:
        return "cirrhosis"
    return "fibrosis"


def is_bt_normal(bt):
    try:
        b = float(bt)
        return 3.0 <= b <= 12.0
    except Exception:
        return False


def bucket_k(k):
    if k is None:
        return 1.5
    choices = [1.0, 1.5, 2.0]
    return min(choices, key=lambda c: abs(c - float(k)))


# ----------------------------------------------------------------------
# J-score interpretation
# Two-pronged: (1) clinical hard rules (APRI/FIB-4 thresholds) take priority,
# (2) otherwise, dynamic tertile classification on the historical cohort.
# ----------------------------------------------------------------------
# Fallback static thresholds — used while the cohort is too small for
# meaningful tertiles. Derived from the v17 dataset range and chosen so the
# "normal" band corresponds to APRI normal × FIB-4 normal × BT normal × K=1.0.
J_SCORE_FALLBACK_THRESHOLDS = (1.0, 5.0)   # (low/mod cut, mod/high cut)
J_SCORE_MIN_COHORT = 20                    # need ≥20 records before using tertiles


def _compute_j_tertiles(db_path):
    """Return (t1, t2) cutoffs from past J-scores, or None if cohort too small."""
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("SELECT j_score FROM submissions WHERE j_score IS NOT NULL")
        vals = [row[0] for row in cur.fetchall() if row[0] is not None]
        con.close()
    except Exception:
        return None
    if len(vals) < J_SCORE_MIN_COHORT:
        return None
    arr = np.asarray(vals, dtype=np.float64)
    t1 = float(np.quantile(arr, 1.0 / 3.0))
    t2 = float(np.quantile(arr, 2.0 / 3.0))
    if t2 <= t1:  # degenerate cohort (all equal)
        return None
    return t1, t2


def _tertile_band(j_score, cutoffs):
    """Return 'low' | 'moderate' | 'high' given (t1, t2) cutoffs."""
    t1, t2 = cutoffs
    if j_score < t1:
        return "low"
    if j_score < t2:
        return "moderate"
    return "high"


def classify_j_score(j_score, db_path):
    """
    Returns a dict:
      {
        "band":     "low" | "moderate" | "high" | None,
        "method":   "tertile" | "fallback" | "no-score",
        "cutoffs":  (t1, t2) | None,
      }
    """
    if j_score is None:
        return {"band": None, "method": "no-score", "cutoffs": None}
    tertiles = _compute_j_tertiles(db_path)
    if tertiles is not None:
        return {"band": _tertile_band(j_score, tertiles),
                "method": "tertile", "cutoffs": tertiles}
    return {"band": _tertile_band(j_score, J_SCORE_FALLBACK_THRESHOLDS),
            "method": "fallback", "cutoffs": J_SCORE_FALLBACK_THRESHOLDS}


def interpret_j_score_key(apri, fib4, bt, k, j_score=None, db_path=None):
    """
    Decide a translation key for the final interpretation.
    Priority:
      1. Hard clinical rules (cirrhosis + abnormal BT → urgent).
      2. J-score band from tertile / fallback classifier.
      3. Fallback to classical v17 rule combinations.
    """
    ca = classify_apri(apri)
    cf = classify_fib4(fib4)
    bt_ok = is_bt_normal(bt)

    # (1) hard clinical override
    if ca == "cirrhosis" and cf == "cirrhosis" and not bt_ok:
        return "interp_urgent"

    # (2) J-score band-based decision
    if j_score is not None and db_path is not None:
        band = classify_j_score(j_score, db_path)["band"]
        if band == "high":
            # Combine with biology to choose urgent vs consult
            if (ca == "cirrhosis" or cf == "cirrhosis") and not bt_ok:
                return "interp_urgent"
            return "interp_consult"
        if band == "moderate":
            return "interp_consult" if (ca in ("fibrosis", "cirrhosis")
                                        or cf in ("fibrosis", "cirrhosis")) \
                                    else "interp_evaluate"
        if band == "low":
            return "interp_normal" if (ca == "normal" and cf == "normal" and bt_ok) \
                                   else "interp_evaluate"

    # (3) classical v17 fallback (no J-score available)
    if ca == "fibrosis" and cf == "fibrosis" and not bt_ok:
        return "interp_consult"
    if ca == "normal" and cf == "normal" and bt_ok:
        return "interp_normal"
    if (ca in ("fibrosis", "cirrhosis")) or (cf in ("fibrosis", "cirrhosis")):
        return "interp_consult"
    return "interp_evaluate"


# ----------------------------------------------------------------------
# Image quality + jaundice index K (preserved from v17)
# ----------------------------------------------------------------------
def assess_photo_quality(image_path):
    im = Image.open(image_path).convert("L").resize((256, 256))
    arr = np.asarray(im, dtype=np.float32) / 255.0
    mean, std = float(arr.mean()), float(arr.std())
    return {
        "too_dark": mean < 0.2,
        "too_bright": mean > 0.9,
        "low_contrast": std < 0.08,
        "reflections": float((arr > 0.98).mean()) > 0.02,
    }


def rough_yellow_ratio(image_path: str) -> float:
    try:
        with Image.open(image_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size
            cw, ch = int(w * 0.6), int(h * 0.6)
            x0, y0 = (w - cw) // 2, (h - ch) // 2
            im = im.crop((x0, y0, x0 + cw, y0 + ch))
            arr = np.asarray(im).astype("float32") / 255.0
            r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
            cmax = arr.max(axis=-1); cmin = arr.min(axis=-1); delta = cmax - cmin + 1e-6
            h_deg = np.where(cmax == r, (60 * ((g - b) / delta) % 360),
                     np.where(cmax == g, 60 * (((b - r) / delta) + 2),
                              60 * (((r - g) / delta) + 3)))
            s = np.where(cmax == 0, 0, delta / cmax)
            v = cmax
            yellow_mask = (h_deg >= 35.0) & (h_deg <= 65.0) & (s >= 0.20) & (v >= 0.50)
            return float(yellow_mask.mean())
    except Exception:
        return 0.0


def compute_jaundice_k_focused(image_path):
    MAX_SIDE = 640; BLUR_RAD = 3; ANGLES = 72
    R_MIN_FRAC, R_MAX_FRAC = 0.06, 0.30; R_STEP = 1.0
    RING_OFFSET = 4; RING_THICK = 10
    S_MIN = 0.20; V_MIN = 0.50; H_Y_MIN, H_Y_MAX = 35.0, 65.0
    K_MIN, K_MAX = 1.0, 3.0

    im = Image.open(image_path)
    im = ImageOps.exif_transpose(im).convert("RGB")
    w0, h0 = im.size
    if max(w0, h0) > MAX_SIDE:
        if w0 >= h0:
            new_w, new_h = MAX_SIDE, int(round(h0 * (MAX_SIDE / w0)))
        else:
            new_h, new_w = MAX_SIDE, int(round(w0 * (MAX_SIDE / h0)))
        im_small = im.resize((new_w, new_h), Image.LANCZOS)
    else:
        im_small = im.copy()
    w, h = im_small.size
    hsv = im_small.convert("HSV")
    H, S, V = [np.asarray(ch, dtype=np.float32) for ch in hsv.split()]
    H_deg = H * (360.0 / 255.0); S_n = S / 255.0; V_n = V / 255.0
    V_blur = np.asarray(im_small.filter(ImageFilter.GaussianBlur(radius=BLUR_RAD)).convert("L"),
                        dtype=np.float32)
    x0, x1 = int(0.2 * w), int(0.8 * w)
    y0, y1 = int(0.2 * h), int(0.8 * h)
    sub = V_blur[y0:y1, x0:x1]
    iy, ix = np.unravel_index(np.argmin(sub), sub.shape)
    cy, cx = iy + y0, ix + x0
    thetas = np.linspace(0, 2 * np.pi, ANGLES, endpoint=False)

    def sample(arr, x, y):
        x = min(max(x, 0), w - 1); y = min(max(y, 0), h - 1)
        x0 = int(np.floor(x)); x1 = min(x0 + 1, w - 1)
        y0 = int(np.floor(y)); y1 = min(y0 + 1, h - 1)
        dx = x - x0; dy = y - y0
        v00 = arr[y0, x0]; v10 = arr[y0, x1]; v01 = arr[y1, x0]; v11 = arr[y1, x1]
        return (v00 * (1 - dx) * (1 - dy) + v10 * dx * (1 - dy)
                + v01 * (1 - dx) * dy + v11 * dx * dy)

    r_min = max(8, int(min(w, h) * R_MIN_FRAC))
    r_max = int(min(w, h) * R_MAX_FRAC)
    radii = np.arange(r_min, r_max, R_STEP, dtype=np.float32)
    ring_mean = []
    for r in radii:
        vals = [sample(V_n, cx + r * np.cos(t), cy + r * np.sin(t)) for t in thetas]
        ring_mean.append(np.mean(vals))
    ring_mean = np.asarray(ring_mean, dtype=np.float32)
    grad = np.zeros_like(ring_mean)
    grad[1:-1] = ring_mean[2:] - ring_mean[:-2]
    best_idx = int(np.argmax(grad))
    r_iris = float(radii[best_idx])

    yy, xx = np.mgrid[0:h, 0:w]
    R = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ring_mask = (R >= (r_iris + RING_OFFSET)) & (R <= (r_iris + RING_OFFSET + RING_THICK))
    ring_mask &= (V_n >= V_MIN)
    valid = ring_mask & (S_n >= S_MIN)
    total = int(np.count_nonzero(valid))
    if total < 50:
        valid = (V_n >= V_MIN) & (S_n >= S_MIN)
        total = int(np.count_nonzero(valid))
    if total == 0:
        fraction_yellow = 0.0
    else:
        yellow = valid & (H_deg >= H_Y_MIN) & (H_deg <= H_Y_MAX)
        fraction_yellow = int(np.count_nonzero(yellow)) / float(total)
    k = float(np.clip(1.0 + 2.0 * fraction_yellow, K_MIN, K_MAX))
    return k, {"fraction_yellow": fraction_yellow}


def opencv_eye_yellow(image_path):
    """Returns (eyes_detected: bool, yellow_pct: float|None)."""
    if cv2 is None:
        return False, None
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False, None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hc = getattr(cv2.data, "haarcascades", None)
        if not hc:
            return False, None
        face_c = cv2.CascadeClassifier(os.path.join(hc, "haarcascade_frontalface_default.xml"))
        eye_c  = cv2.CascadeClassifier(os.path.join(hc, "haarcascade_eye.xml"))
        if face_c.empty() or eye_c.empty():
            return False, None
        faces = face_c.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        rois = []
        for (x, y, w, h) in faces:
            roi_g = gray[y:y + h, x:x + w]
            eyes = eye_c.detectMultiScale(roi_g, 1.15, 7, minSize=(20, 20))
            for (ex, ey, ew, eh) in eyes:
                if ey + eh / 2 < h * 0.7:
                    rois.append((x + ex, y + ey, ew, eh))
        if not rois:
            return False, None
        total = 0; yellow = 0
        for (ex, ey, ew, eh) in rois:
            patch = img[ey:ey + eh, ex:ex + ew]
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (15, 30, 80), (35, 255, 255))
            yellow += int(mask.sum() // 255)
            total  += mask.size
        if total == 0:
            return False, None
        pct = max(0.0, min(100.0, (yellow / float(total)) * 100.0))
        return True, round(pct, 2)
    except Exception:
        return False, None


# ----------------------------------------------------------------------
# PDF report (preserved from v17, with new fields)
# ----------------------------------------------------------------------
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


def generate_pdf(submission_id, data):
    pdf_dir = os.path.join(DATA_FOLDER, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f"submission_{submission_id}.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    w, h = A4

    try:
        logo_path = os.path.join(BASE_DIR, "static", "logo_abdawa.png")
        if os.path.exists(logo_path):
            c.drawImage(logo_path, 1.5 * cm, h - 3.0 * cm,
                        width=1.8 * cm, height=1.8 * cm, mask="auto")
    except Exception:
        pass

    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(w / 2, h - 2.0 * cm, "HepaJScore - Assessment report")
    c.setLineWidth(0.7)
    c.line(1.5 * cm, h - 2.2 * cm, w - 1.5 * cm, h - 2.2 * cm)

    y = h - 3.2 * cm

    def line(t, bold=False):
        nonlocal y
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 11)
        c.drawString(2 * cm, y, str(t))
        y -= 0.72 * cm

    line(f"Patient ID: {data.get('patient_id','')}", bold=True)
    line(f"Photo date: {data.get('date_photo','')}   Biomarker date: {data.get('date_biomarkers','')}")
    line(f"Diagnosis / context: {data.get('diag','')}")
    line(f"Sex: {data.get('sex','')}   Age: {data.get('age','')}")
    y -= 0.2 * cm
    line("Biomarkers", bold=True)
    line(f"ALT: {data.get('alt','')}   AST: {data.get('ast','')}   Platelets: {data.get('plt','')}")
    line(f"Total bilirubin: {data.get('bt','')}   Direct bilirubin: {data.get('bd','')}")
    line(f"Albumin: {data.get('albumin','')}   INR: {data.get('inr','')}   Viral load: {data.get('viral_load','')}")
    y -= 0.2 * cm
    line("Clinical assessment", bold=True)
    line(f"Fibrosis stage: {data.get('fibrosis_stage','')}   Cirrhosis: {data.get('cirrhosis','')}")
    line(f"Ultrasound: {data.get('ultrasound','')}")
    y -= 0.2 * cm
    line("Scores", bold=True)
    jrate = data.get("yellow_ratio", 0.0)
    try:
        jrate_pct = round(max(0.0, min(1.0, float(jrate))) * 100.0, 1)
    except Exception:
        jrate_pct = jrate
    line(f"APRI: {data.get('apri','')}   FIB-4: {data.get('fib4','')}")
    line(f"Jaundice index K: {data.get('k','')}   Detected yellow fraction: {jrate_pct} %")
    line(f"Composite score: {data.get('j_score','')}", bold=True)
    line(f"Interpretation: {data.get('interp','')}", bold=True)

    y -= 0.3 * cm
    c.setFont("Helvetica-Bold", 12)
    line("Informed consent", bold=True)
    c.setFont("Helvetica", 9)
    for t in [
        f"Ethics reference: {CONSENT_REF} (version {CONSENT_VERSION}).",
        "The patient consents to the use of their data (eye photograph and",
        "biomarkers) for research and development purposes.",
    ]:
        line(t)
    y -= 0.2 * cm
    line(f"Signature: {data.get('patient_signature','')}    "
         f"Consent: {'Yes' if data.get('consent') else 'No'}")

    c.setFont("Helvetica-Oblique", 8)
    c.drawCentredString(w / 2, 1.5 * cm,
                        "Research prototype - not a certified medical device. "
                        "Does not replace clinical judgement.")
    c.showPage()
    c.save()
    return pdf_path


# ----------------------------------------------------------------------
# Routes — public
# ----------------------------------------------------------------------
def render_form(lang, **kwargs):
    t = TRANSLATIONS[lang]
    ctx = dict(
        t=t, lang=lang, supported_langs=SUPPORTED_LANGS, translations=TRANSLATIONS,
        today=datetime.date.today().isoformat(), version=VERSION,
        contact_email=CONTACT_EMAIL,
        result=None,
    )
    ctx.update(kwargs)
    return render_template("form.html", **ctx)


@app.route("/", methods=["GET"])
def index():
    return render_form(get_lang())


@app.route("/about", methods=["GET"])
def about():
    lang = get_lang()
    return render_template("about.html", t=TRANSLATIONS[lang], lang=lang,
                           supported_langs=SUPPORTED_LANGS, translations=TRANSLATIONS,
                           version=VERSION, contact_email=CONTACT_EMAIL,
                           consent_ref=CONSENT_REF, consent_version=CONSENT_VERSION)


@app.route("/health", methods=["GET"])
def health():
    try:
        con = sqlite3.connect(DB_PATH); cur = con.cursor()
        cur.execute("SELECT COUNT(*) FROM submissions")
        count = int(cur.fetchone()[0]); con.close()
        return jsonify({"status": "ok", "version": VERSION,
                        "languages": SUPPORTED_LANGS, "db_submissions": count}), 200
    except Exception as e:
        app.logger.exception("health failed")
        return jsonify({"status": "error", "version": VERSION, "error": str(e)}), 500


@app.route("/submit", methods=["POST"])
def submit():
    lang = request.form.get("lang", DEFAULT_LANG)
    if lang not in SUPPORTED_LANGS:
        lang = DEFAULT_LANG
    t = TRANSLATIONS[lang]

    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()
    ua = request.headers.get("User-Agent", "unknown")

    if is_rate_limited(ip):
        flash(t["flash_ratelimit"], "error")
        return render_form(lang)

    # ---- Read fields (all optional except photo) ----
    date_photo_str = request.form.get("date_photo") or ""
    date_bio_str   = request.form.get("date_bio") or ""
    patient_id = (request.form.get("patient_id") or "").strip() or generate_patient_id()
    diag = request.form.get("diag") or ""
    sex  = request.form.get("sex") or ""
    patient_signature = (request.form.get("patient_signature") or "").strip()
    consent = request.form.get("consent") == "on"

    age = to_float(request.form.get("age"))
    alt = to_float(request.form.get("alt"))
    ast = to_float(request.form.get("ast"))
    plt = to_float(request.form.get("plt"))
    afp = to_float(request.form.get("afp"))
    ggt = to_float(request.form.get("ggt"))
    pa  = to_float(request.form.get("pa"))
    bt  = to_float(request.form.get("bt"))
    bd  = to_float(request.form.get("bd"))
    albumin    = to_float(request.form.get("albumin"))
    inr        = to_float(request.form.get("inr"))
    viral_load = to_float(request.form.get("viral_load"))
    uln_ast = to_float(request.form.get("uln_ast")) or 40.0

    fibrosis_stage = (request.form.get("fibrosis_stage") or "").strip()
    cirrhosis      = (request.form.get("cirrhosis") or "").strip()
    ultrasound     = (request.form.get("ultrasound") or "").strip()
    clinical_notes = (request.form.get("clinical_notes") or "").strip()

    # ---- Photo is the ONLY required element ----
    file = request.files.get("eye_photo") or request.files.get("eye_camera")
    if not file or file.filename == "" or not allowed_file(file.filename):
        flash(t["flash_photo_missing"], "error")
        return render_form(lang)

    fname = secure_filename(file.filename)
    timestamp_utc = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    stored = f"{timestamp_utc}_{os.path.splitext(fname)[0]}{os.path.splitext(fname)[1].lower()}"
    img_path = os.path.join(UPLOAD_FOLDER, stored)
    file.save(img_path)

    try:
        with Image.open(img_path) as _im:
            _im.verify()
    except Exception:
        app.logger.exception("Uploaded file is not a valid image")
        try:
            os.remove(img_path)
        except Exception:
            pass
        flash(t["flash_photo_invalid"], "error")
        return render_form(lang)

    # ---- MANDATORY eye detection (refuses image if no eye found) ----
    detection = detect_eyes(img_path)
    if not detection["detected"]:
        # Refuse: no eye in the photo → no computation possible
        app.logger.warning("Eye detection failed for %s: %s",
                           stored, detection.get("reason"))
        try:
            os.remove(img_path)
        except Exception:
            pass
        flash(t["flash_no_eye"], "error")
        return render_form(lang)

    # Yellow fraction limited to the actual sclera mask
    sclera_yellow = sclera_yellow_fraction(img_path, detection)

    q = assess_photo_quality(img_path)
    if any(q.values()):
        flash(t["flash_photo_quality"], "warning")

    try:
        k, kd = compute_jaundice_k_focused(img_path)
    except Exception:
        app.logger.exception("compute_jaundice_k_focused failed")
        k, kd = None, {"fraction_yellow": sclera_yellow}

    # Prefer the sclera-masked yellow fraction over the heuristic K's one
    if sclera_yellow > 0:
        kd["fraction_yellow"] = sclera_yellow
        # Recompute K from sclera-masked yellow fraction for consistency
        k = float(np.clip(1.0 + 2.0 * sclera_yellow, 1.0, 3.0))

    apri = compute_apri(ast, uln_ast, plt)
    fib4 = compute_fib4(age, ast, alt, plt)

    # Eye is confirmed → use the real K
    k_for_score = k if k is not None else 1.0

    j_score = None
    if (apri is not None and fib4 not in (None, 0)
            and bt is not None and k_for_score not in (None, 0)):
        j_score = round((apri / fib4) * (bt * k_for_score), 4)

    interp_key = interpret_j_score_key(apri, fib4, bt, k_for_score,
                                       j_score=j_score, db_path=DB_PATH)
    interp_text = t.get(interp_key, interp_key)

    # ---- Persist ----
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("""
        INSERT INTO submissions
        (timestamp_utc, lang, date_photo, date_biomarkers, patient_id, diag, sex,
         photo_filename, age, alt, ast, plt, afp, ggt, pa, bt, bd, albumin, inr,
         viral_load, uln_ast, fibrosis_stage, cirrhosis, ultrasound, clinical_notes,
         jaundice_k, apri, fib4, j_score, j_interpretation,
         patient_signature, consent, consent_ref, consent_version,
         submission_ip, user_agent, model_version)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (timestamp_utc, lang, date_photo_str, date_bio_str, patient_id, diag, sex,
          stored, age, alt, ast, plt, afp, ggt, pa, bt, bd, albumin, inr,
          viral_load, uln_ast, fibrosis_stage, cirrhosis, ultrasound, clinical_notes,
          k, apri, fib4, j_score, interp_text,
          patient_signature, int(consent), CONSENT_REF, CONSENT_VERSION,
          ip, ua, VERSION))
    sub_id = cur.lastrowid
    con.commit(); con.close()

    app.logger.info("New submission id=%s pid=%s lang=%s j=%s i=%s",
                    sub_id, patient_id, lang, j_score, interp_key)

    pdf_name = None
    try:
        pdf_path = generate_pdf(sub_id, {
            "patient_id": patient_id, "date_photo": date_photo_str,
            "date_biomarkers": date_bio_str, "diag": diag, "sex": sex, "age": age,
            "alt": alt, "ast": ast, "plt": plt, "bt": bt, "bd": bd,
            "albumin": albumin, "inr": inr, "viral_load": viral_load,
            "fibrosis_stage": fibrosis_stage, "cirrhosis": cirrhosis,
            "ultrasound": ultrasound, "apri": apri, "fib4": fib4,
            "k": k, "yellow_ratio": kd.get("fraction_yellow", 0.0),
            "j_score": j_score, "interp": interp_text,
            "patient_signature": patient_signature, "consent": consent,
        })
        pdf_name = os.path.basename(pdf_path)
    except Exception:
        app.logger.exception("pdf failed")

    flash(t["flash_success"], "success")

    j_band_info = classify_j_score(j_score, DB_PATH)

    result = {
        "patient_id": patient_id,
        "j_score": j_score,
        "j_band": j_band_info["band"],          # "low" | "moderate" | "high" | None
        "j_band_method": j_band_info["method"], # "tertile" | "fallback" | "no-score"
        "apri": apri,
        "fib4": fib4,
        "k": k,
        "yellow_ratio": kd.get("fraction_yellow", 0.0),
        "interp_key": interp_key,
        "interp_text": interp_text,
        "pdf_name": pdf_name,
        "detection_method": detection.get("method"),
    }
    return render_form(lang, result=result)


# ----------------------------------------------------------------------
# Static / files
# ----------------------------------------------------------------------
@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    if not require_basic_auth(request.headers.get("Authorization", "")):
        return Response("Authentication required", 401,
                        {"WWW-Authenticate": 'Basic realm="Admin"'})
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=False)


@app.route("/pdfs/<path:filename>")
def serve_pdf(filename):
    return send_from_directory(os.path.join(DATA_FOLDER, "pdfs"),
                               filename, as_attachment=True)


# ----------------------------------------------------------------------
# PWA — manifest + service worker
# ----------------------------------------------------------------------
@app.route("/manifest.webmanifest")
def manifest():
    data = {
        "name": "HepaJScore",
        "short_name": "HepaJScore",
        "description": "Imaging-assisted liver assessment",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#f6f9f5",
        "theme_color": "#134d33",
        "orientation": "portrait",
        "icons": [
            {"src": "/static/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
            {"src": "/static/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"},
        ],
    }
    return Response(json.dumps(data), mimetype="application/manifest+json")


@app.route("/service-worker.js")
def service_worker():
    sw = """
const CACHE = 'hepajscore-v18-1';
const SHELL = ['/', '/about', '/static/styles.css', '/static/app.js', '/static/logo.png', '/manifest.webmanifest'];
self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((keys) =>
    Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
  ).then(() => self.clients.claim()));
});
self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  e.respondWith(
    caches.match(req).then((cached) => cached || fetch(req).then((res) => {
      const copy = res.clone();
      caches.open(CACHE).then((c) => c.put(req, copy)).catch(() => {});
      return res;
    }).catch(() => caches.match('/')))
  );
});
""".strip()
    resp = make_response(sw)
    resp.headers["Content-Type"] = "application/javascript"
    resp.headers["Service-Worker-Allowed"] = "/"
    return resp


# ----------------------------------------------------------------------
# Admin & export (preserved from v17, trilingual shell)
# ----------------------------------------------------------------------
@app.route("/admin", methods=["GET"])
def admin_list():
    if not require_basic_auth(request.headers.get("Authorization", "")):
        return Response("Authentication required", 401,
                        {"WWW-Authenticate": 'Basic realm="Admin"'})
    lang = get_lang()
    q = (request.args.get("q") or "").strip()
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    if q:
        like = f"%{q}%"
        cur.execute("""SELECT id, timestamp_utc, patient_id, j_score, j_interpretation, photo_filename
                       FROM submissions WHERE patient_id LIKE ? OR diag LIKE ?
                       ORDER BY id DESC LIMIT 200""", (like, like))
    else:
        cur.execute("""SELECT id, timestamp_utc, patient_id, j_score, j_interpretation, photo_filename
                       FROM submissions ORDER BY id DESC LIMIT 200""")
    rows = cur.fetchall(); con.close()
    return render_template("admin.html", rows=rows, q=q, version=VERSION,
                           t=TRANSLATIONS[lang], lang=lang,
                           supported_langs=SUPPORTED_LANGS, translations=TRANSLATIONS,
                           contact_email=CONTACT_EMAIL)


@app.route("/admin/<int:sub_id>.json", methods=["GET"])
def admin_detail_json(sub_id):
    if not require_basic_auth(request.headers.get("Authorization", "")):
        return Response("Authentication required", 401,
                        {"WWW-Authenticate": 'Basic realm="Admin"'})
    con = sqlite3.connect(DB_PATH); con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT * FROM submissions WHERE id=?", (sub_id,))
    row = cur.fetchone(); con.close()
    if not row:
        return jsonify({"error": "not found"}), 404
    return jsonify({k: row[k] for k in row.keys()})


@app.route("/admin/delete/<int:sub_id>", methods=["POST", "GET"])
def admin_delete(sub_id):
    if not require_basic_auth(request.headers.get("Authorization", "")):
        return Response("Authentication required", 401,
                        {"WWW-Authenticate": 'Basic realm="Admin"'})
    if not (request.args.get("confirm") == "1" or request.form.get("confirm") == "1"):
        return Response("Add ?confirm=1 to confirm deletion.", 400)
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("SELECT photo_filename FROM submissions WHERE id=?", (sub_id,))
    r = cur.fetchone()
    cur.execute("DELETE FROM submissions WHERE id=?", (sub_id,))
    con.commit(); con.close()
    if r and r[0]:
        try:
            os.remove(os.path.join(UPLOAD_FOLDER, r[0]))
        except Exception:
            pass
    return redirect(url_for("admin_list"))


@app.route("/export/csv", methods=["GET"])
def export_csv():
    if not require_basic_auth(request.headers.get("Authorization", "")):
        return Response("Authentication required", 401,
                        {"WWW-Authenticate": 'Basic realm="Export CSV"'})
    con = sqlite3.connect(DB_PATH); con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT * FROM submissions ORDER BY id DESC")
    rows = cur.fetchall(); con.close()
    output = io.StringIO()
    writer = csv.writer(output)
    if rows:
        writer.writerow(rows[0].keys())
        for r in rows:
            writer.writerow([r[k] for k in r.keys()])
    else:
        writer.writerow(["no data"])
    return Response(output.getvalue(), 200,
                    {"Content-Type": "text/csv; charset=utf-8",
                     "Content-Disposition": "attachment; filename=hepajscore_export.csv"})


@app.route("/admin/print/<int:sub_id>", methods=["GET"])
def admin_print(sub_id):
    if not require_basic_auth(request.headers.get("Authorization", "")):
        return Response("Authentication required", 401,
                        {"WWW-Authenticate": 'Basic realm="Admin"'})
    lang = get_lang()
    con = sqlite3.connect(DB_PATH); con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT * FROM submissions WHERE id=?", (sub_id,))
    row = cur.fetchone(); con.close()
    if not row:
        return Response("Not found", 404)
    data = {k: row[k] for k in row.keys()}
    try:
        k = float(data.get("jaundice_k") or 0)
        frac = max(0.0, min(1.0, (k - 1.0) / 2.0))
        data["yellow_pct"] = round(frac * 100.0, 1)
    except Exception:
        data["yellow_pct"] = None
    return render_template("print.html", d=data,
                           pdf_name=f"submission_{sub_id}.pdf",
                           t=TRANSLATIONS[lang], lang=lang, version=VERSION)


# ----------------------------------------------------------------------
# Error handlers
# ----------------------------------------------------------------------
@app.errorhandler(RequestEntityTooLarge)
def handle_413(e):
    lang = get_lang()
    flash(TRANSLATIONS[lang]["flash_too_large"], "error")
    return render_form(lang), 413


@app.errorhandler(500)
def handle_500(e):
    app.logger.exception("Unhandled 500")
    lang = get_lang()
    flash(TRANSLATIONS[lang]["flash_error"], "error")
    return render_form(lang), 500


# Persist language choice in a cookie on every response
@app.after_request
def set_lang_cookie(resp):
    try:
        lang = request.args.get("lang")
        if lang in SUPPORTED_LANGS:
            resp.set_cookie("hs_lang", lang, max_age=31536000, samesite="Lax")
    except Exception:
        pass
    return resp


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
