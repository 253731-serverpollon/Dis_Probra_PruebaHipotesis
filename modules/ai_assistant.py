# Módulo de asistente IA - versión 1.0
# Integración con Google Gemini para interpretación estadística

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats


def render_ai_assistant():
    st.markdown("## 🤖 Asistente IA — Google Gemini")

    # ── API Key input ──────────────────────────────────────────────────────────
    with st.expander("🔑 Configurar API Key de Google Gemini", expanded="gemini_key" not in st.session_state):
        api_key = st.text_input("Google Gemini API Key", type="password",
                                 help="Obtén tu clave en https://aistudio.google.com/")
        if st.button("Guardar API Key"):
            if api_key.strip():
                st.session_state["gemini_key"] = api_key.strip()
                st.success("✅ API Key guardada.")
            else:
                st.error("La clave no puede estar vacía.")

    if "gemini_key" not in st.session_state:
        st.info("Por favor configura tu API Key de Gemini para continuar.")
        return

    # ── Mode selector ──────────────────────────────────────────────────────────
    st.markdown("### Selecciona el tipo de consulta")
    mode = st.radio("Modo", [
        "📊 Análisis de distribución (datos cargados)",
        "🧪 Interpretación de prueba Z",
        "✏️ Pregunta libre",
    ], label_visibility="collapsed")

    # ──────────────────────────────────────────────────────────────────────────
    if mode == "📊 Análisis de distribución (datos cargados)":
        _mode_distribution()
    elif mode == "🧪 Interpretación de prueba Z":
        _mode_z_test()
    else:
        _mode_free()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _call_gemini(prompt: str) -> str:
    """Call Gemini via google-generativeai SDK."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=st.session_state["gemini_key"])
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error al llamar a Gemini: {e}"

def _show_prompt(prompt: str):
    with st.expander("📋 Prompt enviado a Gemini"):
        st.code(prompt, language="text")


def _show_response(response: str):
    st.markdown(f"""
    <div class="ai-response">
      <div style="font-family:'Space Mono',monospace; font-size:11px; color:#6c63ff; 
                  letter-spacing:2px; margin-bottom:12px;">✦ RESPUESTA GEMINI</div>
      <div style="color:#e8eaf0; font-size:14px; line-height:1.7;">{response.replace(chr(10),'<br>')}</div>
    </div>
    """, unsafe_allow_html=True)


def _mode_distribution():
    st.markdown("#### Análisis de distribución con IA")
    if "data" not in st.session_state:
        st.warning("⚠️ No hay datos cargados. Ve al módulo **Carga de Datos**.")
        return

    data = st.session_state["data"]
    n    = len(data)
    mean = np.mean(data)
    med  = np.median(data)
    s    = np.std(data, ddof=1)
    skew = float(pd.Series(data).skew())
    kurt = float(pd.Series(data).kurtosis())
    q1, q3 = np.percentile(data, [25, 75])
    iqr  = q3 - q1
    n_out = int(np.sum((data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)))

    sw_p = None
    if 8 <= n <= 5000:
        _, sw_p = stats.shapiro(data)

    summary = (
        f"Variable: {st.session_state.get('data_name','N/A')}\n"
        f"n = {n}, Media = {mean:.4f}, Mediana = {med:.4f}, "
        f"Desv. Std = {s:.4f}, Sesgo = {skew:.4f}, "
        f"Curtosis = {kurt:.4f}, Q1 = {q1:.4f}, Q3 = {q3:.4f}, "
        f"IQR = {iqr:.4f}, Outliers (IQR×1.5) = {n_out}"
    )
    if sw_p:
        summary += f", Shapiro-Wilk p = {sw_p:.5f}"

    prompt = (
        "Eres un asistente experto en estadística aplicada. "
        "Analiza el siguiente resumen estadístico y responde en español:\n\n"
        f"{summary}\n\n"
        "Responde:\n"
        "1. ¿La distribución parece normal? Justifica con los estadísticos.\n"
        "2. ¿Hay sesgo? ¿Es relevante?\n"
        "3. ¿Los outliers son problemáticos?\n"
        "4. ¿Qué prueba estadística recomiendas para esta distribución?\n"
        "5. Una conclusión general en 2 oraciones.\n\n"
        "Sé conciso y técnico."
    )

    st.markdown("**Resumen enviado:**")
    st.code(summary, language="text")

    if st.button("🤖 Consultar Gemini", use_container_width=True, key="btn_dist"):
        with st.spinner("Gemini está analizando..."):
            resp = _call_gemini(prompt)
        _show_prompt(prompt)
        _show_response(resp)
        _compare_with_student(data, resp)


def _mode_z_test():
    st.markdown("#### Interpretación de Prueba Z con IA")
    if "z_results" not in st.session_state:
        st.warning("⚠️ Primero ejecuta una prueba Z en el módulo **Prueba Z**.")
        return

    r = st.session_state["z_results"]
    tail_str = {"bilateral": "bilateral (H₁: μ ≠ μ₀)",
                "left":      "cola izquierda (H₁: μ < μ₀)",
                "right":     "cola derecha (H₁: μ > μ₀)"}[r["tail"]]

    summary = (
        f"Media muestral (x̄) = {r['x_bar']:.5f}, "
        f"Media hipotética (μ₀) = {r['mu0']}, "
        f"n = {r['n']}, σ = {r['sigma']}, "
        f"α = {r['alpha']}, tipo = {tail_str}, "
        f"Z calculado = {r['z']:.5f}, "
        f"Valor crítico: {r['region_text']}, "
        f"p-value = {r['p_val']:.5f}"
    )

    prompt = (
        "Eres un asistente experto en inferencia estadística. "
        "Se realizó una prueba Z con los siguientes parámetros:\n\n"
        f"{summary}\n\n"
        "Responde en español:\n"
        "1. ¿Se rechaza H₀? Explica la decisión con base en Z y el p-value.\n"
        "2. ¿Los supuestos de la prueba Z son razonables en este caso?\n"
        "3. ¿Qué significa prácticamente este resultado?\n"
        "4. ¿Alguna limitación o advertencia al interpretar?\n\n"
        "Sé técnico pero claro."
    )

    decision_str = "RECHAZAR H₀" if r["reject"] else "NO RECHAZAR H₀"
    st.markdown(f"**Resumen de la prueba:** `{summary}`")
    st.markdown(f"**Decisión del sistema:** `{decision_str}`")

    if st.button("🤖 Consultar Gemini", use_container_width=True, key="btn_z"):
        with st.spinner("Gemini está analizando..."):
            resp = _call_gemini(prompt)
        _show_prompt(prompt)
        _show_response(resp)

        # Compare decision
        st.markdown("### ⚖️ Comparación: Sistema vs. IA")
        c1, c2 = st.columns(2)
        with c1:
            color = "#ff6b6b" if r["reject"] else "#00d9a0"
            st.markdown(f"""
            <div class="metric-box" style="border-color:{color};">
              <div class="metric-label">Decisión Automática</div>
              <div style="font-family:'Space Mono',monospace; color:{color}; font-size:16px; font-weight:700;">
                {decision_str}
              </div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-box" style="border-color:#6c63ff;">
              <div class="metric-label">Gemini Dice</div>
              <div style="font-family:'DM Sans'; color:#c0c3d4; font-size:13px; margin-top:4px;">
                Ver respuesta arriba →
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:rgba(108,99,255,0.08); border:1px solid #252840;
                    border-radius:8px; padding:14px; margin-top:12px;">
          <b style="font-family:'Space Mono',monospace; color:#6c63ff; font-size:12px;">
            📝 REFLEXIÓN PARA EL ESTUDIANTE
          </b><br>
          <span style="color:#c0c3d4; font-size:13px;">
            Compara la decisión automática con la explicación de Gemini.
            ¿Coinciden? ¿La IA agrega contexto útil? ¿Detectó algún problema con los supuestos?
          </span>
        </div>
        """, unsafe_allow_html=True)


def _mode_free():
    st.markdown("#### Pregunta libre al asistente estadístico")
    user_q = st.text_area("Escribe tu pregunta de estadística:", height=120,
                           placeholder="Ej: ¿Cuándo es apropiado usar una prueba Z vs t? ¿Qué es el error tipo I?")
    if st.button("🤖 Preguntar a Gemini", use_container_width=True, key="btn_free"):
        if not user_q.strip():
            st.warning("Escribe una pregunta primero.")
            return
        prompt = (
            "Eres un asistente experto en estadística y ciencia de datos. "
            "Responde en español de forma clara y técnica:\n\n"
            f"{user_q}"
        )
        with st.spinner("Gemini está respondiendo..."):
            resp = _call_gemini(prompt)
        _show_response(resp)


def _compare_with_student(data, ai_response):
    """Let the student compare their own assessment vs. AI."""
    st.markdown("### ⚖️ ¿Qué opinas tú?")
    st.markdown("""
    <div style="background:rgba(108,99,255,0.08); border:1px dashed #6c63ff;
                border-radius:8px; padding:14px; margin-bottom:12px;">
      <span style="color:#7a7d94; font-size:13px;">
        Reflexiona sobre la respuesta de Gemini. ¿Estás de acuerdo? ¿Detectó algo que tú habías pasado por alto?
        Documenta esto en tu reporte.
      </span>
    </div>
    """, unsafe_allow_html=True)

    skew = float(pd.Series(data).skew())
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    n_out = int(np.sum((data < q1-1.5*iqr) | (data > q3+1.5*iqr)))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Tu evaluación:**")
        student_normal = st.radio("¿La distribución es normal?", ["Sí", "No", "No estoy seguro/a"])
        student_outliers = st.radio("¿Hay outliers problemáticos?", ["Sí", "No"])
    with c2:
        st.markdown("**Análisis automático:**")
        auto_normal = "Sí" if abs(skew) < 0.5 else "No"
        auto_out    = "Sí" if n_out > 0 else "No"
        st.markdown(f"""
        <div style="font-family:'Space Mono',monospace; font-size:13px; color:#c0c3d4; margin-top:28px;">
          Normal: <span style="color:{'#00d9a0' if auto_normal=='Sí' else '#ff6b6b'};">
            {auto_normal} (sesgo={skew:.3f})
          </span><br>
          Outliers: <span style="color:{'#ff6b6b' if auto_out=='Sí' else '#00d9a0'};">
            {auto_out} ({n_out} detectados)
          </span>
        </div>
        """, unsafe_allow_html=True)
