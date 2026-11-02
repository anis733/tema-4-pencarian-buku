"""
Tema 4 - Pencarian Sampul Buku di Rak (UTS Visi Komputer, Kelompok 04)
Pipeline: resize -> grayscale -> ORB/SIFT -> BFMatcher + ratio test -> findHomography (RANSAC)
"""
import csv
import os
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
SAMPLE_DIR = os.path.join(BASE_DIR, "contoh")

# Warna BGR berbeda untuk tiap buku
COLORS = [(0, 200, 0), (255, 120, 0), (0, 0, 255), (200, 0, 200), (0, 200, 255),
          (255, 255, 0), (128, 128, 0), (0, 128, 255)]


# ----------------------------------------------------------------------------
# Fungsi pipeline (dari notebook UTS_Kelompok04.ipynb)
# ----------------------------------------------------------------------------
def resize_image(img, max_side=1000):
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img.copy()
    s = max_side / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def detect_features(gray, method="ORB", nfeatures=5000):
    if method == "ORB":
        det = cv2.ORB_create(nfeatures=nfeatures)
    else:
        det = cv2.SIFT_create(nfeatures=nfeatures)
    return det.detectAndCompute(gray, None)


def match_features(desc_t, desc_s, method="ORB", ratio=0.75):
    if desc_t is None or desc_s is None or len(desc_t) < 2 or len(desc_s) < 2:
        return []
    norm = cv2.NORM_HAMMING if method == "ORB" else cv2.NORM_L2
    bf = cv2.BFMatcher(norm, crossCheck=False)
    good = []
    for pair in bf.knnMatch(desc_t, desc_s, k=2):
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            good.append(pair[0])
    return good


def find_book_in_scene(kp_t, kp_s, good, tpl_shape, min_match=10):
    if len(good) < min_match:
        return False, None, 0
    src = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if M is None:
        return False, None, 0
    inliers = int(mask.sum()) if mask is not None else 0
    if inliers < min_match // 2 or inliers / len(good) < 0.3:
        return False, None, inliers
    h, w = tpl_shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    poly = cv2.perspectiveTransform(corners, M).reshape(4, 2)
    area = cv2.contourArea(poly.astype(np.float32))
    if area < 1000 or area > h * w * 20:
        return False, None, inliers
    if not cv2.isContourConvex(poly.astype(np.int32)):
        return False, None, inliers
    return True, poly, inliers


def draw_box(img, poly, label, color):
    pts = np.int32(poly).reshape(-1, 2)
    cv2.polylines(img, [pts], True, color, 4)
    x, y = int(pts[0][0]), int(pts[0][1])
    y = max(y, 25)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 6, y + 4), color, -1)
    cv2.putText(img, label, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------------
def pil_to_bgr(pil_img):
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@st.cache_data(show_spinner=False)
def load_templates():
    """Muat template bawaan dari folder templates/ -> {nama: gambar BGR}."""
    out = {}
    for f in sorted(os.listdir(TEMPLATE_DIR)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            img = cv2.imread(os.path.join(TEMPLATE_DIR, f))
            if img is not None:
                out[os.path.splitext(f)[0]] = resize_image(img, 1000)
    return out


@st.cache_data(show_spinner=False)
def load_ground_truth():
    gt = {}
    path = os.path.join(SAMPLE_DIR, "ground_truth.csv")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                t = [b.strip() for b in row["buku_target"].split(";") if b.strip()]
                gt[row["file"].strip()] = t
    return gt


def run_search(scene_bgr, templates, method, ratio, min_match, nfeatures, max_side):
    t0 = time.time()
    scene = resize_image(scene_bgr, max_side)
    gray_s = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    kp_s, desc_s = detect_features(gray_s, method, nfeatures)

    result = scene.copy()
    rows, ci = [], 0
    for name, tpl in templates.items():
        gray_t = cv2.cvtColor(resize_image(tpl, max_side), cv2.COLOR_BGR2GRAY)
        kp_t, desc_t = detect_features(gray_t, method, nfeatures)
        good = match_features(desc_t, desc_s, method, ratio)
        found, poly, inl = find_book_in_scene(kp_t, kp_s, good, gray_t.shape, min_match)
        if found:
            draw_box(result, poly, name, COLORS[ci % len(COLORS)])
            ci += 1
        rows.append({"Buku": name, "Status": "✅ Ditemukan" if found else "❌ Tidak Ditemukan",
                     "Good Match": len(good), "Inlier RANSAC": inl})
    return scene, result, rows, time.time() - t0


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Pencarian Sampul Buku di Rak", page_icon="📚", layout="wide")
st.title("📚 Pencarian Sampul Buku di Rak")
st.caption("Tema 4 · UTS Visi Komputer · Kelompok 04 — ORB/SIFT + BFMatcher (ratio test) + Homography RANSAC")

# Sidebar: parameter
with st.sidebar:
    st.header("⚙️ Pengaturan")
    method = st.radio("Metode fitur", ["ORB", "SIFT"], horizontal=True,
                      help="ORB lebih cepat, SIFT lebih akurat untuk sudut miring.")
    ratio = st.slider("Ratio test (Lowe)", 0.5, 0.9, 0.75, 0.05,
                      help="Kecil = lebih ketat, besar = lebih longgar.")
    min_match = st.slider("Minimum good match", 5, 30, 10)
    nfeatures = st.select_slider("Jumlah fitur maks", [1000, 2000, 5000, 8000], value=5000)
    max_side = st.select_slider("Resize sisi terpanjang (px)", [600, 800, 1000, 1200], value=1000)

    st.divider()
    st.subheader("Template buku")
    templates = dict(load_templates())
    use_custom = st.checkbox("Tambah template sendiri")
    if use_custom:
        extra = st.file_uploader("Upload sampul buku (boleh banyak)",
                                 type=["jpg", "jpeg", "png"], accept_multiple_files=True,
                                 key="tpl_up")
        for f in extra or []:
            nm = os.path.splitext(f.name)[0]
            templates[nm] = resize_image(pil_to_bgr(Image.open(f)), 1000)
    st.write(f"{len(templates)} template aktif:")
    cols = st.columns(3)
    for i, (n, im) in enumerate(templates.items()):
        cols[i % 3].image(bgr_to_rgb(im), caption=n, width="stretch")

# Input gambar
sumber = st.radio("Sumber foto rak/meja", ["Upload gambar", "Pakai contoh dataset"], horizontal=True)
scene_bgr, scene_name = None, None

if sumber == "Upload gambar":
    up = st.file_uploader("Upload foto rak/meja berisi buku", type=["jpg", "jpeg", "png"])
    if up is not None:
        scene_bgr = pil_to_bgr(Image.open(up))
        scene_name = up.name
else:
    samples = sorted(f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(".jpg"))
    pick = st.selectbox("Pilih foto contoh", samples)
    scene_bgr = cv2.imread(os.path.join(SAMPLE_DIR, pick))
    scene_name = pick

if scene_bgr is None:
    st.info("Upload foto rak/meja untuk mulai mencari buku.")
    st.stop()

if not templates:
    st.error("Tidak ada template. Tambahkan template di folder `templates/` atau upload di sidebar.")
    st.stop()

with st.spinner("Mencari buku..."):
    scene, result, rows, elapsed = run_search(
        scene_bgr, templates, method, ratio, min_match, nfeatures, max_side)

found_names = [r["Buku"] for r in rows if r["Status"].startswith("✅")]

# Gambar berdampingan
c1, c2 = st.columns(2)
with c1:
    st.subheader("Gambar Asli")
    st.image(bgr_to_rgb(scene), width="stretch")
with c2:
    st.subheader("Gambar Hasil")
    st.image(bgr_to_rgb(result), width="stretch")

# Informasi hasil
st.subheader("Informasi Hasil")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Buku ditemukan", f"{len(found_names)} / {len(templates)}")
m2.metric("Status", "Ditemukan" if found_names else "Tidak Ditemukan")
m3.metric("Metode", method)
m4.metric("Waktu proses", f"{elapsed:.2f} s")

if found_names:
    st.success("Label buku terdeteksi: " + ", ".join(found_names))
else:
    st.warning("Tidak Ditemukan — tidak ada buku target pada foto ini.")

st.dataframe(rows, width="stretch", hide_index=True)

# Bandingkan dengan ground truth (hanya untuk foto contoh dataset)
if sumber == "Pakai contoh dataset":
    gt = load_ground_truth().get(scene_name)
    if gt is not None:
        st.caption("Ground truth: " + (", ".join(gt) if gt else "tidak ada buku target"))
        st.caption("Hasil benar ✅" if set(gt) == set(found_names)
                   else "Hasil berbeda dari ground truth ⚠️")

# Unduh hasil
ok, buf = cv2.imencode(".jpg", result)
if ok:
    st.download_button("⬇️ Unduh gambar hasil", buf.tobytes(),
                       file_name=f"hasil_{scene_name}", mime="image/jpeg")
