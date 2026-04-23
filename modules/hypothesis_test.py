import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import io


ACCENT  = "#6c63ff"
ACCENT2 = "#ff6b6b"
GREEN   = "#00d9a0"
MUTED   = "#7a7d94"


def render_hypothesis_test():
    st.markdown("## 🧪 Prueba Z de Hipótesis")

    if "data" not in st.session_state:
        st.warning("⚠️ Primero carga o genera datos en el módulo **Carga de Datos**.")
        return

    data = st.session_state["data"]
    n    = len(data)
    x_bar = np.mean(data)

    st.markdown(f"""
    <div class="stat-card">
      <span class="tag">SUPUESTOS Z</span>
      <span style="color:#7a7d94; font-size:13px;">
        σ poblacional conocida &nbsp;·&nbsp; n ≥ 30
      </span>
      &nbsp;&nbsp;
      <span class="tag" style="background:{'rgba(0,217,160,0.15)' if n >= 30 else 'rgba(255,107,107,0.15)'};
            color:{'#00d9a0' if n >= 30 else '#ff6b6b'};">
        n = {n} {'✓' if n >= 30 else '✗ (n < 30 — precaución)'}
      </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Parameter panel ────────────────────────────────────────────────────────
    st.markdown("### ⚙️ Parámetros de la prueba")

    col1, col2 = st.columns(2)
    with col1:
        mu0   = st.number_input("Hipótesis nula H₀: μ =", value=round(x_bar, 2), step=0.1,
                                 help="Valor de la media poblacional bajo H₀")
        sigma = st.number_input("Desv. estándar poblacional (σ)", 
                                 value=float(st.session_state.get("sigma_known", np.std(data, ddof=1))),
                                 min_value=0.0001, step=0.1,
                                 help="Si no se conoce σ, usa la desviación muestral como aproximación (n≥30)")
        alpha = st.select_slider("Nivel de significancia (α)", 
                                  options=[0.01, 0.025, 0.05, 0.10],
                                  value=0.05)
    with col2:
        tail = st.radio("Tipo de prueba",
                         ["Bilateral (H₁: μ ≠ μ₀)", 
                          "Cola izquierda (H₁: μ < μ₀)",
                          "Cola derecha (H₁: μ > μ₀)"])

        st.markdown(f"""
        <div style="background:#161924; border:1px solid #252840; border-radius:8px; padding:14px; margin-top:8px;">
          <div style="font-family:'Space Mono',monospace; font-size:12px; color:#7a7d94;">RESUMEN MUESTRA</div>
          <div style="margin-top:8px; font-size:13px; color:#e8eaf0;">
            x̄ = <b style="color:#6c63ff;">{x_bar:.5f}</b><br>
            n = <b style="color:#6c63ff;">{n}</b><br>
            s = <b style="color:#6c63ff;">{np.std(data,ddof=1):.5f}</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Hypothesis display
    tail_key = "bilateral" if "Bilateral" in tail else ("left" if "izquierda" in tail else "right")
    h1_text = {
        "bilateral": f"H₁: μ ≠ {mu0}",
        "left":      f"H₁: μ < {mu0}",
        "right":     f"H₁: μ > {mu0}",
    }[tail_key]

    st.markdown(f"""
    <div style="display:flex; gap:16px; margin:12px 0;">
      <div class="metric-box" style="flex:1;">
        <div class="metric-label">Hipótesis Nula</div>
        <div style="font-family:'Space Mono',monospace; font-size:18px; color:#6c63ff;">H₀: μ = {mu0}</div>
      </div>
      <div class="metric-box" style="flex:1; border-color:#ff6b6b;">
        <div class="metric-label">Hipótesis Alternativa</div>
        <div style="font-family:'Space Mono',monospace; font-size:18px; color:#ff6b6b;">{h1_text}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Run test ───────────────────────────────────────────────────────────────
    if st.button("▶ Ejecutar Prueba Z", use_container_width=True):
        se = sigma / np.sqrt(n)
        z  = (x_bar - mu0) / se

        # Critical values & p-value
        if tail_key == "bilateral":
            z_crit = stats.norm.ppf(1 - alpha / 2)
            p_val  = 2 * (1 - stats.norm.cdf(abs(z)))
            reject = abs(z) > z_crit
            region_text = f"|Z| > {z_crit:.4f}"
        elif tail_key == "left":
            z_crit = stats.norm.ppf(alpha)
            p_val  = stats.norm.cdf(z)
            reject = z < z_crit
            region_text = f"Z < {z_crit:.4f}"
        else:
            z_crit = stats.norm.ppf(1 - alpha)
            p_val  = 1 - stats.norm.cdf(z)
            reject = z > z_crit
            region_text = f"Z > {z_crit:.4f}"

        # ── Store results for AI module ────────────────────────────────────────
        st.session_state["z_results"] = {
            "z": z, "z_crit": z_crit, "p_val": p_val,
            "reject": reject, "tail": tail_key,
            "mu0": mu0, "sigma": sigma, "alpha": alpha,
            "n": n, "x_bar": x_bar, "se": se,
            "region_text": region_text,
        }

        # ── Metrics ────────────────────────────────────────────────────────────
        st.markdown("### 📊 Resultados")
        m1, m2, m3, m4 = st.columns(4)
        for col, (lbl, val) in zip(
            [m1, m2, m3, m4],
            [("Estadístico Z", f"{z:.5f}"),
             ("Valor crítico", region_text),
             ("p-value", f"{p_val:.5f}"),
             ("α", f"{alpha}")]):
            with col:
                st.markdown(f"""
                <div class="metric-box">
                  <div class="metric-label">{lbl}</div>
                  <div class="metric-value" style="font-size:18px;">{val}</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Decision banner ────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        if reject:
            st.markdown(f"""
            <div class="decision-reject">
              ❌ SE RECHAZA H₀ — El estadístico Z = {z:.4f} cae en la región crítica ({region_text}).<br>
              p-value = {p_val:.5f} {'<' if p_val < alpha else '≥'} α = {alpha}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="decision-fail">
              ✅ NO SE RECHAZA H₀ — El estadístico Z = {z:.4f} NO cae en la región crítica ({region_text}).<br>
              p-value = {p_val:.5f} {'<' if p_val < alpha else '≥'} α = {alpha}
            </div>
            """, unsafe_allow_html=True)

        # ── Interpretation ─────────────────────────────────────────────────────
        st.markdown("### 💬 Interpretación")
        if reject:
            interp = (f"Con un nivel de significancia α = {alpha}, existe evidencia estadística "
                      f"suficiente para rechazar la hipótesis nula H₀: μ = {mu0}. "
                      f"El estadístico Z = {z:.4f} se encuentra en la región de rechazo ({region_text}), "
                      f"y el p-value ({p_val:.5f}) es menor que α.")
        else:
            interp = (f"Con un nivel de significancia α = {alpha}, NO existe evidencia estadística "
                      f"suficiente para rechazar la hipótesis nula H₀: μ = {mu0}. "
                      f"El estadístico Z = {z:.4f} no cae en la región crítica ({region_text}), "
                      f"y el p-value ({p_val:.5f}) es mayor o igual que α.")
        st.info(interp)

        # ── Visual: Normal curve with rejection region ─────────────────────────
        st.markdown("### 📉 Curva Normal con Regiones de Rechazo")
        fig = _plot_z_curve(z, z_crit, alpha, tail_key, p_val, reject)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.success("💡 Ve al módulo **Asistente IA** para obtener una explicación detallada de Gemini.")


# ──────────────────────────────────────────────────────────────────────────────

def _plot_z_curve(z_stat, z_crit, alpha, tail, p_val, reject):
    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0d0f17")
    ax.set_facecolor("#0d0f17")

    xs = np.linspace(-4.5, 4.5, 600)
    ys = stats.norm.pdf(xs)

    # Base curve
    ax.plot(xs, ys, color=ACCENT, linewidth=2.5, zorder=3)
    ax.fill_between(xs, ys, alpha=0.08, color=ACCENT)

    # ── Rejection regions ──────────────────────────────────────────────────────
    def fill_reject(from_x, to_x):
        mask = (xs >= from_x) & (xs <= to_x)
        ax.fill_between(xs[mask], ys[mask], color=ACCENT2, alpha=0.55, zorder=2)

    if tail == "bilateral":
        z_c = abs(z_crit)
        fill_reject(-4.5, -z_c)
        fill_reject(z_c, 4.5)
        ax.axvline(-z_c, color=ACCENT2, linewidth=1.5, linestyle="--", alpha=0.8)
        ax.axvline( z_c, color=ACCENT2, linewidth=1.5, linestyle="--", alpha=0.8)
        ax.text(-z_c, -0.012, f"−{z_c:.3f}", ha="center", fontsize=8,
                color=ACCENT2, fontfamily="monospace")
        ax.text( z_c, -0.012, f"+{z_c:.3f}", ha="center", fontsize=8,
                color=ACCENT2, fontfamily="monospace")
    elif tail == "left":
        fill_reject(-4.5, z_crit)
        ax.axvline(z_crit, color=ACCENT2, linewidth=1.5, linestyle="--", alpha=0.8)
        ax.text(z_crit, -0.012, f"{z_crit:.3f}", ha="center", fontsize=8,
                color=ACCENT2, fontfamily="monospace")
    else:  # right
        fill_reject(z_crit, 4.5)
        ax.axvline(z_crit, color=ACCENT2, linewidth=1.5, linestyle="--", alpha=0.8)
        ax.text(z_crit, -0.012, f"{z_crit:.3f}", ha="center", fontsize=8,
                color=ACCENT2, fontfamily="monospace")

    # ── Z statistic line ───────────────────────────────────────────────────────
    z_plot = max(min(z_stat, 4.4), -4.4)
    stat_color = ACCENT2 if reject else GREEN
    ax.axvline(z_plot, color=stat_color, linewidth=2.5, zorder=5)
    arrow_y = stats.norm.pdf(z_plot) + 0.04
    ax.annotate(f"Z = {z_stat:.4f}\np = {p_val:.4f}",
                xy=(z_plot, stats.norm.pdf(z_plot)),
                xytext=(z_plot + (0.6 if z_plot < 0 else -0.6), arrow_y + 0.02),
                fontsize=9, color=stat_color, fontfamily="monospace",
                ha="center",
                arrowprops=dict(arrowstyle="->", color=stat_color, lw=1.5))

    # ── Labels ─────────────────────────────────────────────────────────────────
    ax.set_xlabel("Estadístico Z", fontsize=11)
    ax.set_ylabel("Densidad", fontsize=11)
    ax.set_title(f"Distribución Normal Estándar — Prueba Z ({'bilateral' if tail=='bilateral' else 'una cola'})",
                 fontsize=12, fontweight="bold", pad=12)

    reject_patch  = mpatches.Patch(color=ACCENT2, alpha=0.6, label=f"Región de rechazo (α={alpha})")
    accept_patch  = mpatches.Patch(color=ACCENT,  alpha=0.3, label="Región de no rechazo")
    stat_patch    = plt.Line2D([0],[0], color=stat_color, linewidth=2, label=f"Z calculado = {z_stat:.4f}")
    ax.legend(handles=[reject_patch, accept_patch, stat_patch],
              fontsize=9, loc="upper right")

    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-0.025, 0.46)
    ax.spines[["top","right"]].set_visible(False)

    return fig
