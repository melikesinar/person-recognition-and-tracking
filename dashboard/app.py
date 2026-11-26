"""
Kişi akışı paneli (Faz 4): statistics of a recorded or live session.

Run (from the project root):
    .venv\\Scripts\\streamlit run dashboard/app.py
then open http://localhost:8501 in a browser.

Sessions are written by the system itself (src/__main__.py, tools/replay.py
--record, tools/eval_wildtrack.py --record) to output/sessions/. While the
live system runs, the newest session refreshes every few seconds.
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from analytics import stats  # noqa: E402

SESSIONS_DIR = ROOT / "output" / "sessions"
CAM_COLORS = ["#2563eb", "#ea580c", "#16a34a", "#9333ea", "#db2777", "#0891b2", "#ca8a04"]

st.set_page_config(page_title="Kişi Akışı Paneli", page_icon="👣", layout="wide")


# ------------------------------------------------------------------ helpers

@st.cache_data(show_spinner=False)
def _load(path: str, stamp: float) -> stats.Session:  # stamp: invalidates the cache when files change
    return stats.load_session(Path(path))


def load(path: Path) -> stats.Session:
    stamp = max((f.stat().st_mtime for f in path.glob("*.csv")), default=0.0)
    return _load(str(path), stamp)


def session_label(p: Path) -> str:
    try:
        t = datetime.strptime(p.name[:15], "%Y%m%d_%H%M%S").strftime("%d.%m.%Y %H:%M")
    except ValueError:
        t = p.name
    tag = p.name[16:]
    return f"{t} · {tag}" if tag else f"{t} · canlı"


def fmt_dur(s: float) -> str:
    s = float(s)
    if s < 60:
        return f"{s:.0f} sn"
    if s < 3600:
        return f"{int(s // 60)} dk {int(s % 60):02d} sn"
    return f"{int(s // 3600)} sa {int(s % 3600 // 60):02d} dk"


def time_axis(sess: stats.Session, t: pd.Series) -> pd.Series:
    """Seconds since start for recordings, clock time for live sessions."""
    if sess.is_video:
        return sess.rel(t)
    return pd.to_datetime(t, unit="s", utc=True).dt.tz_convert(datetime.now().astimezone().tzinfo).dt.tz_localize(None)


def fmt_time(sess: stats.Session, t: float) -> str:
    if sess.is_video:
        r = sess.rel(t)
        return f"{int(r // 60):02d}:{r % 60:04.1f}"
    return datetime.fromtimestamp(t).strftime("%H:%M:%S")


def x_enc(sess: stats.Session):
    """X axis: seconds for recordings, clock time (HH:MM:SS) for live sessions."""
    if sess.is_video:
        return alt.X("zaman:Q", title="Kayıt süresi (sn)")
    return alt.X("zaman:T", title="Saat", axis=alt.Axis(format="%H:%M:%S"))


def place_name(sess: stats.Session, place) -> str:
    if isinstance(place, (int, np.integer, float, np.floating)):
        return sess.cam_name(int(place))
    return str(place)


def cam_color_scale(sess: stats.Session):
    names = [c["name"] for c in sess.meta["cameras"]]
    return alt.Scale(domain=names, range=CAM_COLORS[:len(names)])


# ------------------------------------------------------------------ sidebar

st.sidebar.title("👣 Kişi Akışı")
sessions = stats.list_sessions(SESSIONS_DIR)
if not sessions:
    st.title("Kişi Akışı Paneli")
    st.info("Henüz kayıtlı oturum yok. Sistemi çalıştırın (`python src/__main__.py`) ya da bir videoyu "
            "kayıtla işleyin: `python tools/replay.py --cameras config/cameras.epfl-room.yaml --record epfl`.")
    st.stop()

choice = st.sidebar.selectbox("Oturum", sessions, format_func=session_label)
meta_peek = stats.load_session(choice).meta
is_live_candidate = meta_peek.get("clock") == "wall" and choice == sessions[0]
live = st.sidebar.toggle("Canlı yenile (5 sn)", value=is_live_candidate, disabled=not is_live_candidate,
                         help="Sadece sistem şu an çalışıyorsa ve en yeni oturum seçiliyse anlamlı.")
gap = st.sidebar.slider("Ziyaret arası (sn)", 1.0, 30.0, float(meta_peek.get("visit_gap", 5.0)), 1.0,
                        help="Bir kişi bir yerde bu süreden uzun görünmezse oradan ayrılmış sayılır; "
                             "tekrar görünürse yeni ziyaret başlar.")
min_dwell = st.sidebar.slider("En kısa ziyaret (sn)", 0.0, 10.0, 1.0, 0.5,
                              help="Bundan kısa ziyaretler tablolarda gösterilmez (yanından geçip gidenler).")
priv = meta_peek.get("privacy", {})
st.sidebar.caption(
    "🔒 Gizlilik: kişiler sadece numarayla (P1, P2...) tutulur, isim ve fotoğraf yok. "
    f"Kafalar görüntülerde {'pikselleştirilir' if priv.get('heads_pixelated') else 'pikselleştirilmez'}. "
    f"Kayıtlar {priv.get('retention_days', '?')} gün sonra silinir.")


# ------------------------------------------------------------------ page

def render():
    sess = load(choice)
    pos = sess.positions
    sp = float(sess.meta.get("sample_period", 0.5))

    st.title("Kişi Akışı Paneli")
    kind = "kayıt (video)" if sess.is_video else "canlı sistem"
    st.caption(f"Oturum: **{session_label(choice)}** · {kind} · "
               f"kameralar: {', '.join(c['name'] for c in sess.meta['cameras'])}")
    if pos.empty:
        st.warning("Bu oturumda henüz kimse görülmedi.")
        return

    v_cam = stats.visits(pos, "cam", gap, sp)
    v_cam_f = v_cam[v_cam["dwell"] >= min_dwell]
    now = time.time() if (not sess.is_video and live) else sess.last_time
    current = stats.visible_now(pos, now)
    zs = stats.zone_samples(pos, sess.cameras)
    v_zone = stats.visits(zs, "zone", gap, sp)
    lines_ev = stats.line_events(pos, sess.cameras)

    # --- KPIs ---
    k = st.columns(5)
    k[0].metric("Görülen kişi", pos["pid"].nunique())
    k[1].metric("Şu an görünen" if not sess.is_video else "Sonda görünen",
                len({p for ps in current.values() for p in ps}))
    k[2].metric("Ortalama kalış", fmt_dur(v_cam_f["dwell"].mean()) if not v_cam_f.empty else "-")
    busiest = stats.dwell_summary(v_cam_f)
    k[3].metric("En yoğun kamera", place_name(sess, busiest.iloc[0]["place"]) if not busiest.empty else "-")
    k[4].metric("Süre", fmt_dur(pos["t"].max() - pos["t"].min()))

    tabs = st.tabs(["📊 Genel bakış", "🧭 Rotalar", "🔥 Isı haritası", "🟦 Alanlar", "🚪 Giriş-çıkış",
                    "🩺 Sistem sağlığı"])

    # --- Overview ---
    with tabs[0]:
        span = pos["t"].max() - pos["t"].min()
        bin_s = 5 if span < 600 else (60 if span < 4 * 3600 else 600)
        occ = stats.occupancy(pos, bin_s)
        occ["zaman"] = time_axis(sess, occ["bin_start"])
        occ["kamera"] = occ["cam"].map(sess.cam_name)
        st.subheader("Zaman içinde kişi sayısı")
        st.caption(f"Her {bin_s} saniyelik dilimde kamerada görülen farklı kişi sayısı.")
        st.altair_chart(alt.Chart(occ).mark_line(point=True).encode(
            x=x_enc(sess), y=alt.Y("people", title="Kişi"),
            color=alt.Color("kamera", scale=cam_color_scale(sess), title="Kamera"),
            tooltip=["kamera", "people"]).properties(height=280), use_container_width=True)

        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader("Kamera bazında kalış")
            ds = stats.dwell_summary(v_cam_f)
            ds["Kamera"] = ds["place"].map(lambda p: place_name(sess, p))
            st.dataframe(pd.DataFrame({
                "Kamera": ds["Kamera"], "Kişi": ds["people"], "Ziyaret": ds["visits"],
                "Ortalama": ds["avg_s"].map(fmt_dur), "Medyan": ds["median_s"].map(fmt_dur),
                "En uzun": ds["max_s"].map(fmt_dur), "Toplam": ds["total_s"].map(fmt_dur)}),
                hide_index=True, use_container_width=True)
        with c2:
            st.subheader("Şu an kim nerede")
            if current:
                for cam, pids in sorted(current.items()):
                    st.markdown(f"**{sess.cam_name(cam)}** ({len(pids)}): " + ", ".join(f"`P{p}`" for p in pids))
            else:
                st.caption("Son 3 saniyede kimse görünmedi.")

    # --- Routes ---
    with tabs[1]:
        tr = stats.transitions(v_cam_f)
        ts = stats.transition_summary(tr)
        st.subheader("Kameralar arası geçişler")
        if ts.empty:
            overlap = sess.meta.get("topology", {}).get("overlapping")
            st.info("Bu oturumda kameralar arası geçiş yok." +
                    (" Bu kameralar aynı alanı görüyor; bir kişi ikisinde aynı anda görünür, bu bir geçiş sayılmaz."
                     if overlap else ""))
        else:
            ts["Nereden"] = ts["src"].map(sess.cam_name)
            ts["Nereye"] = ts["dst"].map(sess.cam_name)
            c1, c2 = st.columns([2, 3])
            with c1:
                st.altair_chart(alt.Chart(ts).mark_rect().encode(
                    x=alt.X("Nereye:N"), y=alt.Y("Nereden:N"),
                    color=alt.Color("count:Q", title="Geçiş", scale=alt.Scale(scheme="blues")),
                    tooltip=["Nereden", "Nereye", "count", "median_s"]).properties(height=260),
                    use_container_width=True)
            with c2:
                st.dataframe(pd.DataFrame({
                    "Nereden": ts["Nereden"], "Nereye": ts["Nereye"], "Kaç kez": ts["count"],
                    "Medyan süre": ts["median_s"].map(fmt_dur), "En hızlı": ts["min_s"].map(fmt_dur),
                    "En yavaş": ts["max_s"].map(fmt_dur)}), hide_index=True, use_container_width=True)
                st.caption("Süre = bir kameradan çıkış ile diğerinde görünme arası. Topoloji dosyasındaki "
                           "`min_seconds` değerleri buradaki en hızlı sürelere göre ayarlanabilir.")

        st.subheader("Kişi rotaları")
        r = stats.routes(v_cam_f, name=lambda p: sess.cam_name(p), min_dwell=min_dwell)
        st.dataframe(pd.DataFrame({
            "Kişi": r["pid"].map(lambda p: f"P{p}"), "İlk görülme": r["first_seen"].map(lambda t: fmt_time(sess, t)),
            "Son görülme": r["last_seen"].map(lambda t: fmt_time(sess, t)), "Rota": r["route"],
            "Görünür süre": r["time_in_view_s"].map(fmt_dur)}), hide_index=True, use_container_width=True)

    # --- Heat map ---
    with tabs[2]:
        cams = sess.cameras
        cam = st.radio("Kamera", list(cams), format_func=sess.cam_name, horizontal=True)
        bg_path = sess.path / f"background_cam{cam}.jpg"
        bg = cv2.imread(str(bg_path)) if bg_path.is_file() else None
        sub = pos[pos["cam"] == cam]
        fw, fh = cams[cam].get("frame_size") or [int(sub["x"].max() * 1.05) + 1, int(sub["y"].max() * 1.05) + 1]
        if bg is None:
            bg = np.full((int(fh), int(fw), 3), 40, np.uint8)
        scale = bg.shape[1] / fw
        hm = stats.heatmap(pos, cam, (fw, fh), sp, cell=max(4, int(fw) // 90), smooth=1.5)
        if hm.max() > 0:
            norm = hm / hm.max()
            norm_img = cv2.resize(norm.astype(np.float32), (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_CUBIC)
            colored = cv2.applyColorMap(np.uint8(255 * np.clip(norm_img, 0, 1)), cv2.COLORMAP_TURBO)
            alpha = np.clip(norm_img * 1.6, 0, 0.7)[..., None]
            img = (bg * (1 - alpha) + colored * alpha).astype(np.uint8)
        else:
            img = bg.copy()
        for name, poly in (cams[cam].get("zones") or {}).items():
            pts = (np.array(poly) * scale).astype(np.int32)
            cv2.polylines(img, [pts], True, (255, 255, 255), max(1, int(2 * scale)))
            cv2.putText(img, name, tuple(pts.mean(axis=0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4 * max(1, scale), (255, 255, 255), 1, cv2.LINE_AA)
        for name, (a, b) in (cams[cam].get("lines") or {}).items():
            pa, pb = (np.array(a) * scale).astype(int), (np.array(b) * scale).astype(int)
            cv2.line(img, tuple(pa), tuple(pb), (0, 0, 255), max(1, int(2 * scale)))
        c1, c2 = st.columns([3, 1])
        c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
        with c2:
            # Hottest spot: total time in the busiest cell of a coarse grid (~1/30 of the width)
            hot = stats.heatmap(pos, cam, (fw, fh), sp, cell=max(8, int(fw) // 30), smooth=0).max()
            st.metric("En yoğun noktada toplam süre", fmt_dur(hot) if hot > 0 else "-",
                      help="Görüntü genişliğinin yaklaşık 1/30'u büyüklüğündeki en yoğun karede, "
                           "orada bulunan herkesin geçirdiği toplam süre.")
            st.caption("Renkler, insanların **ayaklarının** en çok bulunduğu yerleri gösterir: kırmızı en çok, "
                       "mavi az. Arka plan, birkaç saniye arayla çekilen karelerin ortancası: yürüyen insanlar "
                       "bu yüzden görünmez, kafalar ayrıca pikselleştirilir.")
            st.caption("Beyaz çerçeveler tanımlı alanlar, kırmızı çizgiler sayaç çizgileri.")

    # --- Zones ---
    with tabs[3]:
        if not any(c.get("zones") for c in sess.cameras.values()):
            st.info("Bu oturumda alan tanımlı değil. Kamera dosyasına `zones:` ekleyin "
                    "(`python tools/draw_zones.py --cameras <dosya> --camera <id>` ile çizilir).")
        elif v_zone.empty:
            st.info("Tanımlı alanlarda kimse görülmedi.")
        else:
            vz = v_zone[v_zone["dwell"] >= min_dwell]
            zsum = stats.dwell_summary(vz)
            c1, c2 = st.columns([3, 2])
            with c1:
                st.subheader("Alan bazında kalış")
                st.dataframe(pd.DataFrame({
                    "Alan": zsum["place"], "Kişi": zsum["people"], "Ziyaret": zsum["visits"],
                    "Ortalama": zsum["avg_s"].map(fmt_dur), "Medyan": zsum["median_s"].map(fmt_dur),
                    "En uzun": zsum["max_s"].map(fmt_dur), "Toplam": zsum["total_s"].map(fmt_dur)}),
                    hide_index=True, use_container_width=True)
            with c2:
                st.subheader("Toplam süre")
                st.altair_chart(alt.Chart(zsum).mark_bar().encode(
                    x=alt.X("total_s:Q", title="Saniye"), y=alt.Y("place:N", sort="-x", title=None),
                    tooltip=["place", "people", "total_s"]).properties(height=220), use_container_width=True)
            st.caption("Bir kişi, ayakları alanın içindeyken o alanda sayılır. Aynı alanı iki kamera görüyorsa "
                       "ikisi ayrı satırda görünür.")
            st.subheader("Alanlar arası geçişler")
            # Within one camera only: the same floor seen by two cameras would give fake moves
            ztr = stats.transitions(vz)
            if not ztr.empty:
                ztr = ztr[ztr["src"].str.split(" / ").str[0] == ztr["dst"].str.split(" / ").str[0]]
            zt = stats.transition_summary(ztr)
            if zt.empty:
                st.caption("Alanlar arası geçiş yok.")
            else:
                st.dataframe(pd.DataFrame({"Nereden": zt["src"], "Nereye": zt["dst"], "Kaç kez": zt["count"],
                                           "Medyan süre": zt["median_s"].map(fmt_dur)}),
                             hide_index=True, use_container_width=True)

    # --- Counting lines ---
    with tabs[4]:
        if not any(c.get("lines") for c in sess.cameras.values()):
            st.info("Bu oturumda sayaç çizgisi tanımlı değil. Kamera dosyasına `lines:` ekleyin "
                    "(`tools/draw_zones.py` ile çizilir; ok yönünde geçiş = giriş).")
        else:
            summ = stats.line_summary(lines_ev)
            if summ.empty:
                st.info("Henüz çizgiden geçen olmadı.")
            for row in summ.to_dict("records"):
                st.subheader(f"{row['line']} ({sess.cam_name(row['cam'])})")
                m = st.columns(3)
                m[0].metric("Giriş", int(row["in"]))
                m[1].metric("Çıkış", int(row["out"]))
                m[2].metric("Fark (giriş − çıkış)", int(row["net"]),
                            help="Tek giriş kapısı varsa bu, şu an içeride olan kişi sayısıdır.")
                ev = lines_ev[(lines_ev["cam"] == row["cam"]) & (lines_ev["line"] == row["line"])].copy()
                ev["delta"] = np.where(ev["direction"] == "in", 1, -1)
                ev["içeride"] = ev["delta"].cumsum()
                ev["zaman"] = time_axis(sess, ev["t"])
                st.altair_chart(alt.Chart(ev).mark_line(interpolate="step-after", point=True).encode(
                    x=x_enc(sess),
                    y=alt.Y("içeride", title="Giriş − çıkış"),
                    tooltip=["pid", "direction"]).properties(height=220), use_container_width=True)
            if not lines_ev.empty:
                with st.expander("Tüm geçişler"):
                    st.dataframe(pd.DataFrame({
                        "Zaman": lines_ev["t"].map(lambda t: fmt_time(sess, t)),
                        "Kamera": lines_ev["cam"].map(sess.cam_name), "Çizgi": lines_ev["line"],
                        "Kişi": lines_ev["pid"].map(lambda p: f"P{p}"),
                        "Yön": lines_ev["direction"].map({"in": "giriş", "out": "çıkış"})}),
                        hide_index=True, use_container_width=True)

    # --- Health ---
    with tabs[5]:
        h = sess.health
        if h.empty:
            st.info("Bu oturumda sağlık kaydı yok. Sağlık bilgisi (işleme hızı, bağlantı durumu) canlı "
                    "sistemde 10 saniyede bir yazılır; video kayıtlarında bu sekme boştur.")
        else:
            h = h.astype({"t": float, "cam": int, "fps": float, "target_fps": float})
            h["kamera"] = h["cam"].map(sess.cam_name)
            h["zaman"] = time_axis(sess, h["t"])
            last = h.sort_values("t").groupby("cam").tail(1)
            cols = st.columns(max(1, len(last)))
            icons = {"ok": "🟢 bağlı", "stalled": "🟠 görüntü gelmiyor", "reconnecting": "🔴 yeniden bağlanıyor",
                     "failed": "🔴 açılamadı"}
            stale = (time.time() - last["t"].max()) > 30 if not sess.is_video else False
            for col, row in zip(cols, last.itertuples()):
                col.metric(row.kamera, f"{row.fps:.1f} / {row.target_fps:g} kare/sn",
                           help="Gerçek işleme hızı / hedef hız")
                col.caption(icons.get(row.status, row.status) + (" (sistem kapalı olabilir)" if stale else ""))
            st.altair_chart(alt.Chart(h).mark_line().encode(
                x=x_enc(sess), y=alt.Y("fps", title="İşlenen kare/sn"),
                color=alt.Color("kamera", scale=cam_color_scale(sess), title="Kamera")).properties(height=240),
                use_container_width=True)
            bad = h[h["status"] != "ok"]
            if not bad.empty:
                st.warning(f"{len(bad)} ölçümde bağlantı sorunu görüldü.")


st.fragment(render, run_every=5 if live else None)()
