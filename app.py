from pathlib import Path
import pandas as pd
import joblib
import streamlit as st
from urllib.parse import urlparse
import difflib
import io

# ==== Yollar ====
ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "phishing_url_baseline.joblib"

# ==== Eğitim kodundaki yardımcılar (birebir) ====
def _registered_domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if ":" in host:
        host = host.split(":")[0]
    parts = [p for p in host.split(".") if p]
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host

_POPULAR = [
    "google.com", "youtube.com", "facebook.com", "apple.com",
    "amazon.com", "twitter.com", "x.com", "github.com",
]

def _looks_like_typosquat(url: str, similarity_thr: float = 0.80) -> bool:
    dom = _registered_domain(url)
    for ref in _POPULAR:
        if dom == ref:
            return False
        if difflib.SequenceMatcher(None, dom, ref).ratio() >= similarity_thr:
            return True
    return False

def _extra_suspicious_signals(url: str) -> bool:
    scheme = urlparse(url).scheme.lower()
    dom = _registered_domain(url)
    has_digits_in_domain = any(ch.isdigit() for ch in dom.split(".")[0])
    return (scheme == "http") or has_digits_in_domain

@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model dosyası yok: {MODEL_FILE}\nÖnce train_baseline.py dosyasını çalıştırıp modeli kaydedin."
        )
    return joblib.load(MODEL_FILE)

def predict_one(url: str, base_thr: float = 0.50):
    pipe = load_model()
    proba = float(pipe.predict_proba([url])[:, 1][0])
    pred = int(proba >= base_thr)
    reasons = []

    # Kural ayarlamaları
    if pred == 0 and (_looks_like_typosquat(url) or _extra_suspicious_signals(url)):
        pred = 1
        reasons.append("Kural: typosquat/HTTP/sayılı-domain sinyali")

    # Açıklama amaçlı zenginleştirme
    dom = _registered_domain(url)
    if urlparse(url).scheme.lower() == "http":
        reasons.append("HTTP (TLS yok)")
    if any(ch.isdigit() for ch in dom.split(".")[0]):
        reasons.append("Alan adı ilk kısmında sayı var")
    if _looks_like_typosquat(url):
        reasons.append(f"Popüler domaine çok benzer: {dom}")

    label = "PHISHING" if pred == 1 else "LEGIT"
    return label, proba, ", ".join(sorted(set(reasons))) if reasons else "-"

def predict_batch(urls, base_thr: float = 0.50) -> pd.DataFrame:
    pipe = load_model()
    urls = [str(u) for u in urls]
    probs = pipe.predict_proba(urls)[:, 1]
    preds = (probs >= base_thr).astype(int)

    adjusted = []
    reasons_all = []
    for u, yhat in zip(urls, preds):
        reasons = []
        final = int(yhat)
        if final == 0 and (_looks_like_typosquat(u) or _extra_suspicious_signals(u)):
            final = 1
            reasons.append("Kural: typosquat/HTTP/sayılı-domain")
        dom = _registered_domain(u)
        if urlparse(u).scheme.lower() == "http":
            reasons.append("HTTP (TLS yok)")
        if any(ch.isdigit() for ch in dom.split(".")[0]):
            reasons.append("Alan adı ilk kısmında sayı var")
        if _looks_like_typosquat(u):
            reasons.append(f"Popüler domaine çok benzer: {dom}")
        adjusted.append(final)
        reasons_all.append(", ".join(sorted(set(reasons))) if reasons else "-")

    df = pd.DataFrame({
        "url": urls,
        "pred_proba": probs,
        "pred_label": adjusted
    })
    df["pred_text"] = df["pred_label"].map({1: "PHISHING", 0: "LEGIT"})
    df["reasons"] = reasons_all
    return df[["url", "pred_text", "pred_proba", "reasons"]]

# ==== UI ====
st.set_page_config(page_title="Phishing URL Tespiti", page_icon="🛡️", layout="centered")
st.title("🛡️ Phishing URL Tespiti (Baseline)")

with st.sidebar:
    st.markdown("**Eşik (proba ≥ eşik ⇒ PHISHING)**")
    thr = st.slider("Eşik", 0.0, 1.0, 0.50, 0.01)
    st.caption("Not: Eşik düştükçe recall artar, precision düşebilir.")

tab1, tab2 = st.tabs(["Tek URL", "Toplu Skorlama (CSV/Text)"])

with tab1:
    url = st.text_input("URL girin", placeholder="https://example.com/login")
    if st.button("Tahmin Et", type="primary") and url:
        try:
            label, proba, why = predict_one(url, base_thr=thr)
            st.metric(label="Sonuç", value=label, delta=f"proba={proba:.3f}")
            st.write("**Gerekçeler:**", why)
        except Exception as e:
            st.error(str(e))

with tab2:
    st.write("Aşağıya her satıra bir URL yapıştırın **veya** `url` kolonu olan bir CSV yükleyin.")
    txt = st.text_area("Çoklu URL", height=150, placeholder="https://a...\nhttp://b...\nhttps://c...")
    up = st.file_uploader("CSV yükle (isteğe bağlı)", type=["csv"])
    run = st.button("Toplu Skorla", type="primary")

    if run:
        try:
            urls = []
            if txt.strip():
                urls.extend([line.strip() for line in txt.splitlines() if line.strip()])
            if up is not None:
                df_in = pd.read_csv(up)
                if "url" not in df_in.columns:
                    st.error("CSV içinde 'url' kolonu yok.")
                else:
                    urls.extend(df_in["url"].astype(str).tolist())

            urls = list(dict.fromkeys(urls))  # tekrarsız
            if not urls:
                st.warning("Skorlanacak URL bulunamadı.")
            else:
                out = predict_batch(urls, base_thr=thr)
                st.dataframe(out, use_container_width=True)
                # indirilebilir CSV
                buf = io.StringIO()
                out.to_csv(buf, index=False)
                st.download_button(
                    "CSV indir",
                    data=buf.getvalue(),
                    file_name="scored_urls.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(str(e))

st.caption("Model: TF-IDF (char 2–5-gram) + Logistic Regression, kural eklemeleriyle.")
