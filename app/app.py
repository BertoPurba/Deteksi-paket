from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# -------------------- Konfigurasi --------------------
st.set_page_config(
    page_title="PackGuard AI | DATASCAPE 2026",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_NAMES = ["Aman", "Lainnya", "Rusak"]
IMG_SIZE = (224, 224)

# Mendukung app.py di root project maupun satu folder di bawah root.
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILENAMES = ["efficientnetb0_final.keras", "efficientnetb0_best.keras"]
MODEL_CANDIDATES = [
    folder / filename
    for folder in (BASE_DIR / "models", BASE_DIR.parent / "models", BASE_DIR.parent.parent / "models", BASE_DIR)
    for filename in MODEL_FILENAMES
]
MODEL_PATH = next((p.resolve() for p in MODEL_CANDIDATES if p.exists()), None)

# ikon, warna status (diambil dari tema), saran tindak lanjut
CLASS_META = {
    "Aman": ("✅", "ok", "Paket tampak utuh. Dapat dilanjutkan ke proses berikutnya."),
    "Lainnya": ("⚠️", "warn", "Gambar tidak termasuk kondisi paket standar. Periksa ulang foto atau lakukan pemeriksaan manual."),
    "Rusak": ("🚫", "bad", "Terindikasi ada kerusakan. Disarankan pemeriksaan manual sebelum paket diproses."),
}

# -------------------- Tema (terang / gelap) --------------------
THEMES = {
    "light": {
        "ink": "#14213D", "muted": "#64748B", "line": "#E4EAF2",
        "paper": "#F4F6FA", "card": "#FFFFFF", "accent": "#1E5EFF",
        "track": "#E8EDF5", "dash": "#B8C4D8",
        "ok": "#17803F", "ok-bg": "#E6F6EC",
        "warn": "#B45F06", "warn-bg": "#FFF3E0",
        "bad": "#C0322B", "bad-bg": "#FDECEA",
        "hero-a": "#14213D", "hero-b": "#21458C",
        "shadow": "0 1px 2px rgba(20,33,61,.04), 0 10px 28px rgba(20,33,61,.07)",
    },
    "dark": {
        "ink": "#E8EEFA", "muted": "#8D9BB5", "line": "#222C45",
        "paper": "#0A0F1D", "card": "#111A2E", "accent": "#5B8CFF",
        "track": "#1C2740", "dash": "#34436A",
        "ok": "#4ADE80", "ok-bg": "#10281C",
        "warn": "#FBBF4D", "warn-bg": "#2F2410",
        "bad": "#FF8A80", "bad-bg": "#33181A",
        "hero-a": "#101B36", "hero-b": "#1B3A7A",
        "shadow": "0 1px 2px rgba(0,0,0,.4), 0 12px 32px rgba(0,0,0,.4)",
    },
}

st.session_state.setdefault("dark_mode", False)
is_dark = st.session_state["dark_mode"]
T = THEMES["dark" if is_dark else "light"]

theme_vars = "".join(f"--{k}: {v};" for k, v in T.items())
st.markdown(
    f"""<style>
:root {{ {theme_vars} color-scheme: {"dark" if is_dark else "light"}; }}
</style>""",
    unsafe_allow_html=True,
)

# -------------------- Tampilan --------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
.stApp { background: var(--paper); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stHeader"] * { color: var(--ink); }
[data-testid="stToolbar"], #MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1200px; }

/* Paksa warna teks mengikuti tema aplikasi */
.stApp p, .stApp span, .stApp label, .stApp li,
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] p,
[data-testid="stCaptionContainer"], [data-testid="stSpinner"] { color: var(--ink); }
[data-testid="stAlert"] p, [data-testid="stAlert"] div { color: var(--ink); }
hr { border-color: var(--line) !important; }
h1, h2, h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -.03em; color: var(--ink); }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--card); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] li, [data-testid="stSidebar"] span { color: var(--ink); }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: var(--muted); }
.brand { display: flex; align-items: center; gap: .65rem; font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 1.2rem; letter-spacing: -.03em; }
.brand-logo { width: 36px; height: 36px; border-radius: 10px; background: var(--accent); display: grid; place-items: center; font-size: 1.1rem; }
.legend { display: flex; gap: .7rem; align-items: flex-start; margin-bottom: .8rem; font-size: .9rem; line-height: 1.4; }
.dot { width: 10px; height: 10px; border-radius: 50%; margin-top: .4rem; flex: none; }
.legend small { display: block; color: var(--muted); }
.small-note { color: var(--muted) !important; font-size: .8rem; line-height: 1.5; }
[data-testid="stSidebar"] .small-note { color: var(--muted) !important; }

/* Hero */
.eyebrow { text-transform: uppercase; letter-spacing: .16em; font-size: .72rem; font-weight: 700; color: var(--muted); }
.hero {
    position: relative; overflow: hidden;
    background: linear-gradient(120deg, var(--hero-a) 0%, var(--hero-b) 100%);
    padding: 2.3rem 2.4rem; border-radius: 26px; margin: .9rem 0 1.4rem; box-shadow: var(--shadow);
    border: 1px solid rgba(255,255,255,.06);
}
.hero::after {
    content: ""; position: absolute; right: -80px; top: -90px; width: 320px; height: 320px; border-radius: 50%;
    background: radial-gradient(circle, rgba(120,160,255,.35), transparent 70%);
}
.hero .eyebrow { color: #9DBAFF; }
.hero h1 { font-family: 'Manrope', sans-serif; color: #fff !important; font-size: clamp(1.9rem, 3.2vw, 2.8rem); line-height: 1.1; letter-spacing: -.045em; margin: .7rem 0; }
.hero p { color: #D7E1F1 !important; font-size: 1rem; max-width: 560px; margin: 0 0 1.2rem; }
.hero-tags { display: flex; flex-wrap: wrap; gap: .5rem; position: relative; z-index: 1; }
.hero-tags span { font-size: .78rem; font-weight: 600; color: #E8EFFF !important; background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.16); padding: .3rem .7rem; border-radius: 999px; }

/* Judul langkah */
.step { display: flex; align-items: center; gap: .7rem; margin: .2rem 0 .3rem; }
.step-n { width: 30px; height: 30px; border-radius: 9px; background: var(--accent); color: #fff !important; font-weight: 700; font-size: .85rem; display: grid; place-items: center; }
.step-t { font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 1.3rem; letter-spacing: -.03em; }
.step-d { color: var(--muted) !important; font-size: .92rem; margin-bottom: .9rem; }

/* Panel */
.panel { background: var(--card); border: 1px solid var(--line); border-radius: 20px; padding: 1.3rem; box-shadow: var(--shadow); }
.empty { text-align: center; color: var(--muted); padding: 2rem 0; }
.empty .ico { font-size: 2.3rem; }
.empty b { color: var(--ink); }

/* Uploader */
[data-testid="stFileUploaderDropzone"] { background: var(--card); border: 1.5px dashed var(--dash); border-radius: 18px; padding: 1.6rem; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent); }
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] p { color: var(--muted) !important; }
[data-testid="stFileUploaderDropzone"] button { background: var(--card); border: 1px solid var(--line); border-radius: 10px; }
[data-testid="stFileUploaderDropzone"] button *,
[data-testid="stFileUploaderFile"] * { color: var(--ink) !important; }

/* Gambar */
[data-testid="stImage"] img { border-radius: 18px; border: 1px solid var(--line); box-shadow: var(--shadow); }
[data-testid="stImageCaption"] { color: var(--muted); font-size: .85rem; }

/* Tombol */
div.stButton > button { border-radius: 12px; min-height: 3rem; font-weight: 700; transition: transform .1s ease; }
div.stButton > button[kind="primary"] { background: var(--accent); border: 0; }
div.stButton > button[kind="primary"] * { color: #fff !important; }
div.stButton > button[kind="primary"]:hover { filter: brightness(1.08); }
div.stButton > button[kind="primary"]:disabled { opacity: .4; }
div.stButton > button[kind="secondary"] { background: var(--card); border: 1px solid var(--line); }
div.stButton > button[kind="secondary"] * { color: var(--ink) !important; }
div.stButton > button:active { transform: scale(.98); }

/* Expander (detail error) */
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 12px; background: var(--card); }
[data-testid="stExpander"] * { color: var(--ink); }

/* Hasil */
.verdict { border-radius: 20px; padding: 1.4rem 1.5rem; border: 1px solid var(--c); background: var(--bg); box-shadow: var(--shadow); }
.v-kicker { font-size: .74rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; color: var(--c) !important; }
.v-main { display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin: .5rem 0 .2rem; }
.v-label { font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 2.6rem; letter-spacing: -.04em; line-height: 1; color: var(--c) !important; display: flex; align-items: center; gap: .6rem; }
.v-conf { text-align: right; line-height: 1.1; }
.v-conf b { font-family: 'Manrope', sans-serif; font-size: 1.8rem; font-weight: 800; color: var(--c) !important; }
.v-conf span { display: block; font-size: .75rem; color: var(--muted) !important; }
.v-advice { margin-top: .9rem; font-size: .93rem; line-height: 1.5; color: var(--ink) !important; }

.prob-title { font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 1.05rem; letter-spacing: -.02em; margin: 0 0 .9rem; }
.prob-row { margin-bottom: .85rem; }
.prob-row:last-child { margin-bottom: 0; }
.prob-head { display: flex; justify-content: space-between; font-size: .9rem; font-weight: 600; margin-bottom: .35rem; }
.prob-head span:last-child { font-variant-numeric: tabular-nums; color: var(--muted); font-weight: 500; }
.prob-row.top .prob-head span:last-child { color: var(--ink); font-weight: 700; }
.prob-track { height: 9px; border-radius: 999px; background: var(--track); overflow: hidden; }
.prob-fill { height: 100%; border-radius: 999px; }

@media (max-width: 640px) {
    .hero { padding: 1.5rem; }
    .v-label { font-size: 2rem; }
    .v-main { flex-direction: column; align-items: flex-start; }
    .v-conf { text-align: left; }
}
@media (prefers-reduced-motion: no-preference) { .prob-fill { transition: width .5s ease; } }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------- Fungsi model --------------------
@st.cache_resource(show_spinner="Memuat model klasifikasi...")
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path, compile=False)


def prepare_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    # Pipeline training memakai nilai piksel 0–255; jangan dinormalisasi ke 0–1 di sini.
    image_array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(image_array, axis=0)


def render_verdict(result: dict):
    icon, tone, advice = CLASS_META[result["label"]]
    st.markdown(
        f"""<div class="verdict" style="--c:var(--{tone});--bg:var(--{tone}-bg)">
<div class="v-kicker">Predicted condition</div>
<div class="v-main">
<div class="v-label"><span>{icon}</span>{result["label"]}</div>
<div class="v-conf"><b>{result["confidence"]:.1%}</b><span>confidence score</span></div>
</div>
<div class="v-advice">{advice}</div>
</div>""",
        unsafe_allow_html=True,
    )


def render_probabilities(probabilities: list):
    order = np.argsort(probabilities)[::-1]
    rows = []
    for rank, i in enumerate(order):
        name = CLASS_NAMES[i]
        tone = CLASS_META[name][1]
        pct = float(probabilities[i]) * 100
        top = " top" if rank == 0 else ""
        rows.append(
            f'<div class="prob-row{top}"><div class="prob-head"><span>{name}</span><span>{pct:.1f}%</span></div>'
            f'<div class="prob-track"><div class="prob-fill" style="width:{pct:.2f}%;background:var(--{tone})"></div></div></div>'
        )
    st.markdown(
        '<div class="panel"><div class="prob-title">Probability per class</div>' + "".join(rows) + "</div>",
        unsafe_allow_html=True,
    )


def step_header(number: str, title: str, desc: str):
    st.markdown(
        f'<div class="step"><div class="step-n">{number}</div><div class="step-t">{title}</div></div>'
        f'<div class="step-d">{desc}</div>',
        unsafe_allow_html=True,
    )


# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-logo">📦</div>PackGuard AI</div>', unsafe_allow_html=True)
    st.caption("DATASCAPE 2026 · Computer Vision")
    st.toggle("🌙 Mode gelap", key="dark_mode")
    st.divider()
    st.markdown("**Tentang aplikasi**")
    st.write("Klasifikasi kondisi fisik paket menggunakan model EfficientNetB0.")
    st.markdown("**Kategori**")
    st.markdown(
        '<div class="legend"><div class="dot" style="background:var(--ok)"></div><div><b>Aman</b><small>Paket tampak utuh</small></div></div>'
        '<div class="legend"><div class="dot" style="background:var(--warn)"></div><div><b>Lainnya</b><small>Bukan kondisi paket standar</small></div></div>'
        '<div class="legend"><div class="dot" style="background:var(--bad)"></div><div><b>Rusak</b><small>Paket tampak mengalami kerusakan</small></div></div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown('<p class="small-note">Hasil AI adalah bantuan pemeriksaan awal, bukan pengganti pemeriksaan manual.</p>', unsafe_allow_html=True)

# -------------------- Header --------------------
st.markdown('<div class="eyebrow">DATASCAPE 2026 / INTELLIGENT INSPECTION</div>', unsafe_allow_html=True)
st.markdown(
    """<div class="hero">
<div class="eyebrow">PACKAGE QUALITY · AI VISION</div>
<h1>Inspect packages.<br>Make better decisions.</h1>
<p>Unggah foto paket dan biarkan model AI memperkirakan kondisinya dalam beberapa detik.</p>
<div class="hero-tags"><span>EfficientNetB0</span><span>224 × 224 RGB</span><span>3 kelas</span></div>
</div>""",
    unsafe_allow_html=True,
)

if MODEL_PATH is None:
    st.error(
        "File model tidak ditemukan. Letakkan `efficientnetb0_final.keras` atau "
        "`efficientnetb0_best.keras` di folder `models/` pada root project."
    )
    st.code("datascape/\n├── app/\n│   └── app.py\n└── models/\n    └── efficientnetb0_final.keras", language="text")
    st.stop()

# Memuat model hanya sekali selama sesi aplikasi.
try:
    model = load_model(str(MODEL_PATH))
except Exception as error:
    st.error("Model gagal dimuat. Periksa versi TensorFlow dan file model.")
    with st.expander("Detail error"):
        st.code(str(error))
    st.stop()

left, right = st.columns([1.02, 0.98], gap="large")

with left:
    step_header("1", "Upload package image", "Gunakan foto yang jelas dengan kondisi paket terlihat. Format: JPG, JPEG, PNG, WEBP, BMP.")
    uploaded_file = st.file_uploader(
        "Pilih gambar paket",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )
    preview_image = None
    file_id = None
    if uploaded_file is not None:
        try:
            preview_image = Image.open(uploaded_file).convert("RGB")
            file_id = f"{uploaded_file.name}-{uploaded_file.size}"
            st.image(preview_image, caption=uploaded_file.name, use_container_width=True)
        except Exception:
            st.error("Gambar tidak dapat dibaca. Coba gunakan file gambar lain.")
    else:
        st.markdown(
            '<div class="panel"><div class="empty"><div class="ico">↥</div><b>Belum ada gambar</b>'
            '<p class="small-note">Pilih gambar di area upload untuk memulai pemeriksaan.</p></div></div>',
            unsafe_allow_html=True,
        )

    predict_clicked = st.button(
        "Analyze package condition →",
        type="primary",
        use_container_width=True,
        disabled=preview_image is None,
    )

with right:
    step_header("2", "Inspection result", "Kondisi paket dan skor probabilitas tiap kelas.")

    if predict_clicked and preview_image is not None:
        with st.spinner("AI sedang memeriksa gambar..."):
            try:
                input_array = prepare_image(preview_image)
                probabilities = model.predict(input_array, verbose=0)[0]
                predicted_index = int(np.argmax(probabilities))
                predicted_label = CLASS_NAMES[predicted_index]
                confidence = float(probabilities[predicted_index])
                st.session_state["last_result"] = {
                    "label": predicted_label,
                    "confidence": confidence,
                    "probabilities": probabilities.tolist(),
                    "filename": uploaded_file.name,
                    "file_id": file_id,
                }
            except Exception as error:
                st.error("Prediksi gagal dijalankan.")
                with st.expander("Detail error"):
                    st.code(str(error))

    result = st.session_state.get("last_result")
    # Tampilkan hanya hasil milik gambar yang sedang diunggah.
    if result and result.get("file_id") == file_id:
        render_verdict(result)
        st.write("")
        if result["confidence"] < 0.6:
            st.warning("Confidence rendah. Coba foto yang lebih tajam dan terang, dengan seluruh paket terlihat.")
        render_probabilities(result["probabilities"])
        st.caption(f"File diperiksa: {result['filename']}")
        st.markdown(
            '<p class="small-note">Confidence adalah probabilitas keluaran model, bukan jaminan bahwa prediksi benar.</p>',
            unsafe_allow_html=True,
        )
        if st.button("Clear result", use_container_width=True):
            st.session_state.pop("last_result", None)
            st.rerun()
    else:
        st.markdown(
            '<div class="panel" style="min-height:300px;display:flex;align-items:center;justify-content:center">'
            '<div class="empty"><div class="ico">◉</div>'
            '<b>Waiting for image analysis</b><p class="small-note">Hasil prediksi dan skor tiap kelas akan muncul di sini.</p></div></div>',
            unsafe_allow_html=True,
        )

st.divider()
footer_left, footer_right = st.columns([2, 1])
with footer_left:
    st.markdown("**PackGuard AI** · Package Condition Classification")
    st.markdown('<p class="small-note">Dibuat untuk DATASCAPE 2026. Gunakan gambar yang relevan dengan data pelatihan agar hasil lebih bermakna.</p>', unsafe_allow_html=True)
with footer_right:
    st.markdown('<p class="small-note" style="text-align:right">Model: EfficientNetB0<br>Input: 224 × 224 RGB<br>Classes: 3</p>', unsafe_allow_html=True)