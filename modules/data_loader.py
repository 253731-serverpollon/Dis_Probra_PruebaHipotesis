import streamlit as st
import pandas as pd
import numpy as np


def render_data_loader():
    st.markdown("## 📂 Carga de Datos")

    tab1, tab2 = st.tabs(["📁 Cargar CSV", "🎲 Datos Sintéticos"])

    # ── Tab 1: CSV ─────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("""
        <div class="stat-card">
        <b style="font-family:'Space Mono',monospace; color:#6c63ff;">CARGAR ARCHIVO CSV</b><br>
        <span style="color:#7a7d94; font-size:13px;">El archivo debe contener columnas numéricas para análisis estadístico.</span>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader("Selecciona un archivo CSV", type=["csv"])
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                st.success(f"✅ Archivo cargado: **{uploaded.name}** — {df.shape[0]} filas × {df.shape[1]} columnas")

                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    st.error("❌ No se encontraron columnas numéricas en el archivo.")
                else:
                    col_sel = st.selectbox("Selecciona la variable a analizar:", numeric_cols)
                    st.session_state["data"] = df[col_sel].dropna().values
                    st.session_state["data_name"] = col_sel
                    st.session_state["data_source"] = f"CSV: {uploaded.name} — columna: {col_sel}"

                    st.dataframe(df[numeric_cols].describe().round(4), use_container_width=True)
                    st.info(f"🎯 Variable seleccionada: **{col_sel}** — n = {len(st.session_state['data'])}")
            except Exception as e:
                st.error(f"Error al leer el archivo: {e}")

    # ── Tab 2: Synthetic ───────────────────────────────────────────────────────
    with tab2:
        st.markdown("""
        <div class="stat-card">
        <b style="font-family:'Space Mono',monospace; color:#6c63ff;">GENERADOR DE DATOS SINTÉTICOS</b><br>
        <span style="color:#7a7d94; font-size:13px;">Genera muestras con distribuciones conocidas para experimentar.</span>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            dist_type = st.selectbox("Distribución", [
                "Normal", "Sesgada (Log-Normal)", "Uniforme",
                "Exponencial", "Bimodal"
            ])
            n = st.slider("Tamaño de muestra (n)", 30, 2000, 200, 10)
        with c2:
            mean_val = st.number_input("Media (μ)", value=50.0, step=1.0)
            std_val  = st.number_input("Desv. estándar (σ)", value=10.0, min_value=0.1, step=0.5)

        seed = st.number_input("Semilla aleatoria", value=42, step=1)

        if st.button("🎲 Generar Datos", use_container_width=True):
            rng = np.random.default_rng(int(seed))
            if dist_type == "Normal":
                data = rng.normal(mean_val, std_val, n)
                label = "Normal"
            elif dist_type == "Sesgada (Log-Normal)":
                data = rng.lognormal(np.log(mean_val), 0.6, n)
                label = "Log-Normal (sesgada)"
            elif dist_type == "Uniforme":
                data = rng.uniform(mean_val - std_val * 1.73, mean_val + std_val * 1.73, n)
                label = "Uniforme"
            elif dist_type == "Exponencial":
                data = rng.exponential(std_val, n) + (mean_val - std_val)
                label = "Exponencial"
            else:  # Bimodal
                d1 = rng.normal(mean_val - std_val, std_val * 0.5, n // 2)
                d2 = rng.normal(mean_val + std_val, std_val * 0.5, n - n // 2)
                data = np.concatenate([d1, d2])
                label = "Bimodal"

            st.session_state["data"]        = data
            st.session_state["data_name"]   = label
            st.session_state["data_source"] = f"Sintético — {label} | μ={mean_val}, σ={std_val}, n={n}"
            st.session_state["sigma_known"] = std_val  # useful default for Z-test

            st.success(f"✅ Datos generados: **{label}** — n = {n}")
            _show_quick_stats(data)

    # ── Preview if data loaded ─────────────────────────────────────────────────
    if "data" in st.session_state:
        st.markdown("---")
        st.markdown("### 🔍 Vista rápida")
        _show_quick_stats(st.session_state["data"])


def _show_quick_stats(data: np.ndarray):
    mean  = np.mean(data)
    med   = np.median(data)
    s     = np.std(data, ddof=1)
    skew  = float(pd.Series(data).skew())
    kurt  = float(pd.Series(data).kurtosis())
    n     = len(data)

    cols = st.columns(6)
    metrics = [
        ("n",        f"{n:,}"),
        ("Media",    f"{mean:.4f}"),
        ("Mediana",  f"{med:.4f}"),
        ("Desv. Std",f"{s:.4f}"),
        ("Sesgo",    f"{skew:.4f}"),
        ("Curtosis", f"{kurt:.4f}"),
    ]
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-box">
              <div class="metric-label">{label}</div>
              <div class="metric-value" style="font-size:20px;">{value}</div>
            </div>
            """, unsafe_allow_html=True)
