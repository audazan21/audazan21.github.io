from pathlib import Path
import pandas as pd
import joblib
import streamlit as st
import difflib, io, re, ipaddress
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "phishing_url_baseline.joblib"

_PSL2 = {
    "com.tr","net.tr","org.tr","edu.tr","gov.tr","mil.tr","bel.tr","pol.tr","k12.tr","gen.tr","av.tr","bbs.tr","name.tr",
    "co.uk","org.uk","ac.uk","gov.uk","sch.uk","ltd.uk","plc.uk","me.uk",
    "com.au","net.au","org.au","edu.au","gov.au",
    "co.jp","ne.jp","or.jp","ac.jp","go.jp",
    "co.nz","org.nz","ac.nz","govt.nz"
}

_SAFE_SUBS = {
    "www","m","mail","webmail","owa","smtp","imap","pop","pop3","mx","ns","cdn","static","img","images","files","file",
    "drive","dl","download","api","s","gw","vpn","id","login","auth","cas","sso","portal","webvpn"
}

def _refang(u: str) -> str:
    s = (u or "").strip()
    s = s.replace("hxxps://", "https://").replace("hxxp://", "http://")
    s = s.replace("[.]", ".").replace("(.)", ".").replace("{.}", ".")
    s = s.replace("[dot]", ".").replace("(dot)", ".")
    s = re.sub(r"\s+", "", s)
    return s

def _normalize_url(u: str) -> str:
    s = _refang(u)
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", s):
        s = "http://" + s
    return s

def _split_host(host: str):
    labels = [p for p in host.split(".") if p]
    sfx_len = 2 if len(labels) >= 2 and ".".join(labels[-2:]) in _PSL2 else 1
    return labels, sfx_len

def _registered_domain(url: str) -> str:
    p = urlparse(_normalize_url(url))
    host = (p.hostname or "").lower()
    if ":" in host:
        host = host.split(":")[0]
    labels, sfx_len = _split_host(host)
    if len(labels) >= 1 + sfx_len:
        return ".".join(labels[-(1 + sfx_len):])
    return host

def _is_ip_host(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False

def _brandless_anomaly_flags(url: str):
    p = urlparse(_normalize_url(url))
    raw_host = p.netloc
    host = (p.hostname or "").lower()
    pathq = (p.path or "") + ("?" + p.query if p.query else "")
    flags, notes = [], []
    if p.scheme.lower() == "http":
        flags.append("http")
    if p.port and p.port not in (80, 443):
        flags.append("nonstd-port"); notes.append(f"port={p.port}")
    if "@" in raw_host or p.username or p.password:
        flags.append("userinfo@")
    if _is_ip_host(host):
        flags.append("ip-host")
    if "xn--" in host:
        flags.append("punycode")
    if re.search(r"[^\x00-\x7f]", host):
        flags.append("non-ascii")
        if re.search(r"[a-z]", host):
            flags.append("mixed-script")
    labels, sfx_len = _split_host(host)
    subs = labels[:-(1 + sfx_len)] if len(labels) >= 1 + sfx_len else []
    sld  = labels[-(1 + sfx_len)] if len(labels) >= 1 + sfx_len else (labels[0] if labels else "")
    noisy_subs = [l for l in subs if not re.fullmatch(r"www\d*", l) and l not in _SAFE_SUBS]
    if len(noisy_subs) >= 3:
        flags.append("many-subdomains"); notes.append(f"subdepth={len(noisy_subs)}")
    check_labels = [sld] + noisy_subs
    for lab in check_labels:
        if any(ch.isdigit() for ch in lab):
            flags.append("digit-in-label")
        if re.search(r"[a-z][0-9]|[0-9][a-z]", lab):
            flags.append("digit-subst")
        if re.search(r"(.)\1\1", lab) and not re.fullmatch(r"www\d*", lab):
            flags.append("repeat-run")
        if len(lab) >= 15:
            flags.append("long-label")
        if lab.startswith("-") or lab.endswith("-"):
            flags.append("hyphen-edge")
    if re.search(r"login|verify|billing|update|confirm|secure|recovery", pathq, re.I):
        flags.append("phishy-kw")
    if len(url) >= 100:
        flags.append("long-url")
    return flags, notes

_POPULAR = [
    "google.com","youtube.com","facebook.com","apple.com",
    "amazon.com","twitter.com","x.com","github.com"
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
    p = urlparse(_normalize_url(url))
    scheme = p.scheme.lower()
    host = (p.hostname or "") 
    sld = host.split(".")[0] if host else ""
    has_digits_in_domain = any(ch.isdigit() for ch in sld)
    return (scheme == "http") or has_digits_in_domain

@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model dosyası yok: {MODEL_FILE}\nÖnce train_baseline.py dosyasını çalıştırıp modeli kaydedin.")
    return joblib.load(MODEL_FILE)

def predict_one(url: str, base_thr: float = 0.50, use_rules: bool = True):
    url = _normalize_url(url)
    pipe = load_model()
    proba = float(pipe.predict_proba([url])[:, 1][0])
    dom = _registered_domain(url)
    flags = []
    if _looks_like_typosquat(url):
        flags.append("typosquat")
    if urlparse(url).scheme.lower() == "http":
        flags.append("http")
    sld = dom.split(".")[0] if dom else ""
    if any(ch.isdigit() for ch in sld):
        flags.append("sayılı-domain")
    flags2, notes2 = _brandless_anomaly_flags(url)
    flags.extend(flags2)
    final_score = proba
    if use_rules:
        boost = {
            "userinfo@": 0.95,
            "punycode": 0.92,
            "mixed-script": 0.90,
            "ip-host": 0.85,
            "typosquat": 0.75,
            "digit-subst": 0.75,
            "digit-in-label": 0.72,
            "many-subdomains": 0.70,
            "nonstd-port": 0.70,
            "phishy-kw": 0.70,
            "sayılı-domain": 0.70,
            "long-url": 0.68,
            "long-label": 0.68,
            "hyphen-edge": 0.68,
            "non-ascii": 0.72,
            "repeat-run": 0.72
        }
        for f in flags:
            if f in boost:
                final_score = max(final_score, boost[f])
    pred = int(final_score >= base_thr)
    reasons = []
    if flags:
        reasons.append("Kural (tetiklenen): " + ", ".join(sorted(set(flags))))
    if notes2:
        reasons.extend(notes2)
    try:
        best_ref, best_sim = None, 0.0
        for ref in _POPULAR:
            r = difflib.SequenceMatcher(None, dom, ref).ratio()
            if r > best_sim:
                best_ref, best_sim = ref, r
        if best_ref and best_ref != dom and best_sim >= 0.80:
            reasons.append(f"Popüler domaine çok benzer: {best_ref} (sim={best_sim:.2f})")
    except NameError:
        pass
    label = "PHISHING" if pred == 1 else "LEGIT"
    return label, proba, ", ".join(reasons) if reasons else "-"

def predict_batch(urls, base_thr: float = 0.50, use_rules: bool = True) -> pd.DataFrame:
    urls = [_normalize_url(str(u)) for u in urls]
    rows = []
    for u in urls:
        label, proba, why = predict_one(u, base_thr=base_thr, use_rules=use_rules)
        rows.append((u, label, proba, why))
    df = pd.DataFrame(rows, columns=["url", "pred_text", "pred_proba", "reasons"])
    return df[["url", "pred_text", "pred_proba", "reasons"]]

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
            label, proba, why = predict_one(url, base_thr=thr, use_rules=True)
            st.metric(label="Sonuç", value=label, delta=f"proba={proba:.3f}")
            st.write("**Gerekçeler:**", why)
        except Exception as e:
            st.error(str(e))

with tab2:
    st.write("Aşağıya her satıra bir URL yapıştırın veya `url` kolonu olan bir CSV yükleyin.")
    txt = st.text_area("Çoklu URL", height=150, placeholder="https://a...\nhttp://b...\nhttps://c...")
    up = st.file_uploader("CSV yükle (isteğe bağlı)", type=["csv"])
    run = st.button("Toplu Skorla", type="primary")
    if run:
        try:
            urls = []
            if txt.strip():
                urls.extend([_normalize_url(line.strip()) for line in txt.splitlines() if line.strip()])
            if up is not None:
                df_in = pd.read_csv(up)
                if "url" not in df_in.columns:
                    st.error("CSV içinde 'url' kolonu yok.")
                else:
                    urls.extend([_normalize_url(u) for u in df_in["url"].astype(str).tolist()])
            urls = list(dict.fromkeys(urls))
            if not urls:
                st.warning("Skorlanacak URL bulunamadı.")
            else:
                out = predict_batch(urls, base_thr=thr, use_rules=True)
                st.dataframe(out, use_container_width=True)
                buf = io.StringIO()
                out.to_csv(buf, index=False)
                st.download_button("CSV indir", data=buf.getvalue(), file_name="scored_urls.csv", mime="text/csv")
        except Exception as e:
            st.error(str(e))

st.caption("Model: TF-IDF (char 2–5-gram) + Logistic Regression, kural eklemeleriyle.")

