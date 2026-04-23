import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import io


ACCENT  = "#6c63ff"
ACCENT2 = "#ff6b6b"
GREEN   = "#00d9a0"
MUTED   = "#7a7d94"


def render_visualizations():
    st.markdown("## 📈 Visualización de Distribuciones")

    if "data" not in st.session_state:
        st.warning("⚠️ Primero carga o genera datos en el módulo **Carga de Datos**.")
        return

    data = st.session_state["data"]
    name = st.session_state.get("data_name", "Variable")

    st.markdown(f"""
    <div class="stat-card">
      <span class="tag">FUENTE</span>
      <span style="color:#7a7d94; font-size:13px;">{st.session_state.get('data_source','—')}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Controls ───────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        n_bins = st.slider("Bins del histograma", 10, 100, 30)
    with c2:
        show_kde    = st.checkbox("Mostrar KDE", value=True)
    with c3:
        show_normal = st.checkbox("Superponer curva normal", value=True)

    # ── Build figure ───────────────────────────────────────────────────────────
    fig = _build_figure(data, name, n_bins, show_kde, show_normal)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Automatic interpretation ───────────────────────────────────────────────
    st.markdown("### 🔍 Interpretación Automática")
    _interpret_distribution(data, name)

    # ── Normality tests ────────────────────────────────────────────────────────
    st.markdown("### 🧪 Pruebas de Normalidad")
    _normality_tests(data)


# ──────────────────────────────────────────────────────────────────────────────

def _build_figure(data, name, n_bins, show_kde, show_normal):
    fig = plt.figure(figsize=(14, 10), facecolor="#0d0f17")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax_hist = fig.add_subplot(gs[0, :])   # full-width histogram
    ax_box  = fig.add_subplot(gs[1, 0])   # boxplot
    ax_qq   = fig.add_subplot(gs[1, 1])   # QQ plot

    # ── 1. Histogram ──────────────────────────────────────────────────────────
    counts, bins, patches = ax_hist.hist(
        data, bins=n_bins, density=True,
        color=ACCENT, alpha=0.65, edgecolor="#0d0f17", linewidth=0.4,
        label="Histograma"
    )
    if show_kde:
        kde   = stats.gaussian_kde(data)
        xs    = np.linspace(data.min(), data.max(), 300)
        ax_hist.plot(xs, kde(xs), color=ACCENT2, linewidth=2.5, label="KDE")
    if show_normal:
        mu, sigma = np.mean(data), np.std(data)
        xs = np.linspace(data.min(), data.max(), 300)
        ax_hist.plot(xs, stats.norm.pdf(xs, mu, sigma),
                     color=GREEN, linewidth=2, linestyle="--", label="Normal teórica")
    ax_hist.axvline(np.mean(data),   color=GREEN,   linewidth=1.5, linestyle=":", alpha=0.9, label=f"Media={np.mean(data):.2f}")
    ax_hist.axvline(np.median(data), color="#ffd93d", linewidth=1.5, linestyle=":", alpha=0.9, label=f"Mediana={np.median(data):.2f}")
    ax_hist.set_title(f"Distribución — {name}", fontsize=13, fontweight="bold", pad=12)
    ax_hist.set_xlabel("Valor", fontsize=11)
    ax_hist.set_ylabel("Densidad", fontsize=11)
    ax_hist.legend(fontsize=9, loc="upper right")

    # ── 2. Boxplot ─────────────────────────────────────────────────────────────
    bp = ax_box.boxplot(
        data, vert=True, patch_artist=True,
        flierprops=dict(marker="o", color=ACCENT2, markersize=4, alpha=0.6),
        medianprops=dict(color=GREEN, linewidth=2),
        boxprops=dict(facecolor=f"{ACCENT}33", edgecolor=ACCENT, linewidth=1.5),
        whiskerprops=dict(color=MUTED, linewidth=1.5),
        capprops=dict(color=MUTED, linewidth=1.5),
    )
    ax_box.set_title("Boxplot", fontsize=12, fontweight="bold")
    ax_box.set_ylabel("Valor", fontsize=10)
    ax_box.set_xticks([])

    # Annotate quartiles
    q1, med, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    for val, lbl in [(q1, "Q1"), (med, "Mediana"), (q3, "Q3")]:
        ax_box.text(1.18, val, f"{lbl}\n{val:.2f}", va="center",
                    fontsize=8, color=MUTED, fontfamily="monospace")

    # ── 3. QQ Plot ─────────────────────────────────────────────────────────────
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")
    ax_qq.scatter(osm, osr, color=ACCENT, s=15, alpha=0.7, label="Datos")
    line_x = np.array([min(osm), max(osm)])
    ax_qq.plot(line_x, slope * line_x + intercept,
               color=GREEN, linewidth=2, linestyle="--", label=f"Línea teórica (r={r:.3f})")
    ax_qq.set_title("Q-Q Plot (Normalidad)", fontsize=12, fontweight="bold")
    ax_qq.set_xlabel("Cuantiles teóricos", fontsize=10)
    ax_qq.set_ylabel("Cuantiles observados", fontsize=10)
    ax_qq.legend(fontsize=9)

    return fig


def _interpret_distribution(data, name):
    n      = len(data)
    mean   = np.mean(data)
    median = np.median(data)
    s      = np.std(data, ddof=1)
    skew   = float(pd.Series(data).skew())
    kurt   = float(pd.Series(data).kurtosis())
    q1, q3 = np.percentile(data, [25, 75])
    iqr    = q3 - q1
    lower  = q1 - 1.5 * iqr
    upper  = q3 + 1.5 * iqr
    n_out  = np.sum((data < lower) | (data > upper))

    # Normal?
    sw_stat, sw_p = stats.shapiro(data[:5000]) if n >= 8 else (None, None)
    normal_flag   = (sw_p is not None and sw_p > 0.05) or (abs(skew) < 0.5 and abs(kurt) < 1)

    lines = []

    # Normality
    if normal_flag:
        lines.append(("✅ Distribución normal",
                       f"El sesgo ({skew:.3f}) y la curtosis ({kurt:.3f}) sugieren que la distribución "
                       f"es aproximadamente normal. La media ({mean:.3f}) y la mediana ({median:.3f}) son cercanas.",
                       "ok"))
    else:
        lines.append(("⚠️ Distribución no normal",
                       f"El sesgo ({skew:.3f}) indica {'asimetría positiva (cola derecha)' if skew > 0 else 'asimetría negativa (cola izquierda)'}. "
                       f"Curtosis = {kurt:.3f}.",
                       "warn"))

    # Outliers
    if n_out == 0:
        lines.append(("✅ Sin outliers detectados",
                       f"No hay valores fuera de [{lower:.2f}, {upper:.2f}] (criterio IQR×1.5).",
                       "ok"))
    else:
        lines.append((f"⚠️ {n_out} outlier(s) detectado(s)",
                       f"Hay {n_out} valor(es) fuera de [{lower:.2f}, {upper:.2f}]. "
                       f"Representa el {100*n_out/n:.1f}% de la muestra.",
                       "warn"))

    # Skew detail
    if abs(skew) > 1:
        lines.append(("📐 Sesgo pronunciado",
                       f"Sesgo = {skew:.3f}. {'Cola larga a la derecha.' if skew > 0 else 'Cola larga a la izquierda.'}",
                       "warn"))

    for title, desc, typ in lines:
        color = "#00d9a0" if typ == "ok" else "#ff6b6b"
        st.markdown(f"""
        <div style="background:{'rgba(0,217,160,0.07)' if typ=='ok' else 'rgba(255,107,107,0.07)'};
                    border-left:3px solid {color}; border-radius:6px;
                    padding:14px 18px; margin:8px 0;">
          <b style="color:{color}; font-family:'Space Mono',monospace; font-size:13px;">{title}</b><br>
          <span style="color:#c0c3d4; font-size:13px;">{desc}</span>
        </div>
        """, unsafe_allow_html=True)


def _normality_tests(data):
    n = len(data)
    results = []

    # Shapiro-Wilk (n ≤ 5000)
    if 8 <= n <= 5000:
        stat, p = stats.shapiro(data)
        results.append(("Shapiro-Wilk", stat, p, "W"))

    # Kolmogorov-Smirnov
    stat_ks, p_ks = stats.kstest(data, "norm",
                                  args=(np.mean(data), np.std(data)))
    results.append(("Kolmogorov-Smirnov", stat_ks, p_ks, "D"))

    # D'Agostino K²
    if n >= 20:
        stat_da, p_da = stats.normaltest(data)
        results.append(("D'Agostino K²", stat_da, p_da, "K²"))

    rows = []
    for name, stat, p, label in results:
        decision = "✅ No rechaza normalidad" if p > 0.05 else "❌ Rechaza normalidad"
        rows.append({"Prueba": name, f"Estadístico ({label})": f"{stat:.5f}",
                     "p-value": f"{p:.5f}", "Decisión (α=0.05)": decision})

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
