# 📊 StatLab AI — Aplicación de Análisis Estadístico

Aplicación interactiva en **Streamlit** para visualización de distribuciones,
pruebas de hipótesis Z y asistente de IA con Google Gemini.

---

## 🚀 Instalación y ejecución

```bash
# 1. Clona el repositorio
git clone https://github.com/TU_USUARIO/statlab-ai.git
cd statlab-ai

# 2. Crea entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Instala dependencias
pip install -r requirements.txt

# 4. Ejecuta la app
streamlit run app.py
```

La app abre automáticamente en http://localhost:8501

---

## 📦 Módulos

| Módulo | Descripción |
|--------|-------------|
| `📂 Carga de Datos` | Sube CSV o genera datos sintéticos (Normal, Log-Normal, Uniforme, Exponencial, Bimodal) |
| `📈 Visualizaciones` | Histograma + KDE, Boxplot, Q-Q Plot, pruebas de normalidad automáticas |
| `🧪 Prueba Z` | Prueba Z (σ conocida, n≥30): bilateral, cola izquierda, cola derecha. Curva con zona de rechazo. |
| `🤖 Asistente IA` | Integración con Google Gemini para interpretación estadística |

---

## 🔑 API Key de Gemini

1. Ve a https://aistudio.google.com/
2. Crea una API Key gratuita
3. Ingrésala en el módulo **Asistente IA** de la app

---

## 📐 Supuestos de la Prueba Z implementada

- **σ poblacional conocida** (o se usa s muestral como aproximación cuando n≥30)
- **n ≥ 30** (Teorema Central del Límite garantiza aproximación normal)
- Tipos: bilateral, cola izquierda, cola derecha

---

## 📁 Estructura del proyecto

```
statlab-ai/
├── app.py                    # Punto de entrada principal
├── requirements.txt          # Dependencias
├── README.md
└── modules/
    ├── __init__.py
    ├── data_loader.py         # Módulo de carga de datos
    ├── visualizations.py      # Histograma, KDE, Boxplot, QQ
    ├── hypothesis_test.py     # Prueba Z + curva de rechazo
    └── ai_assistant.py        # Integración Gemini
```

---

## 📋 Entregables del proyecto

- [ ] Video Parte A: Proceso de desarrollo
- [ ] Video Parte B: Demo de uso de la app
- [ ] Reporte de desarrollo (ver plantilla)
- [ ] Repositorio GitHub con commits distribuidos

---

## ⚠️ Notas importantes para el estudiante

- Los commits deben estar **distribuidos en el tiempo** (no todo en una sesión)
- Cada commit debe tener **mensaje descriptivo** (ej: `feat: add Z-test bilateral logic`)
- Documenta tu interacción con herramientas de IA en el reporte
