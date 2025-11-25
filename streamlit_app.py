import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np

# ==============================
#  Configuración general (Tema Colegio Unión de Ñaña)
# ==============================
st.set_page_config(
    page_title="Panel de Aprobación – Colegio Unión de Ñaña",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
<style>
:root {
    --primary: #003c71;          /* azul institucional */
    --primary-soft: #e3edf7;
    --accent: #facc15;           /* dorado suave */
    --accent-soft: #fef9c3;
    --bg-page: #f3f4f6;
    --bg-card: #ffffff;
    --border-subtle: #e5e7eb;
    --text-main: #111827;
    --text-muted: #6b7280;
}

/* Fondo general claro */
main, .stApp {
    background: var(--bg-page);
}

/* Tipografía general */
html, body, .stApp, .stMarkdown, p, li, span, label,
h1, h2, h3, h4, h5, h6, .stCaption {
    color: var(--text-main);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Tabs tipo pastilla */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    padding: 0.4rem 1.1rem;
    background-color: #e5e7eb;
    color: #374151;
    font-weight: 500;
    border: 1px solid #d1d5db;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, var(--primary), #1d4ed8) !important;
    color: #f9fafb !important;
    border-color: transparent !important;
}

/* Tarjeta principal */
.card {
    background-color: var(--bg-card);
    border-radius: 1rem;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 18px 35px rgba(15, 23, 42, 0.08);
    border: 1px solid var(--border-subtle);
}

/* Tarjetas de métricas */
.metric-card {
    border-radius: 0.9rem;
    padding: 0.9rem 1.1rem;
    background: linear-gradient(135deg, var(--primary-soft), #eef2ff);
    color: var(--text-main);
    border: 1px solid #bfdbfe;
}
.metric-card.pass {
    background: linear-gradient(135deg, var(--accent-soft), #fffbeb);
    border-color: #fde68a;
}
.metric-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: var(--text-muted);
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 750;
}
.metric-sub {
    font-size: 0.9rem;
    color: var(--text-muted);
}

/* Botón principal */
.stButton>button {
    border-radius: 999px;
    background: linear-gradient(90deg, #22c55e, #16a34a);
    border: none;
    color: #f9fafb;
    font-weight: 700;
    padding: 0.55rem 1.9rem;
    box-shadow: 0 10px 24px rgba(34, 197, 94, 0.35);
}
.stButton>button:hover {
    filter: brightness(1.05);
}

/* Selects claros */
.stSelectbox > div > div {
    background-color: #ffffff !important;
    color: var(--text-main) !important;
    border-radius: 0.75rem !important;
    border: 1px solid #d1d5db !important;
}
.stSelectbox svg {
    color: var(--text-muted) !important;
}

/* Menú de select */
div[data-baseweb="select"] ul {
    background-color: #ffffff !important;
}
div[data-baseweb="select"] li {
    color: var(--text-main) !important;
}
div[data-baseweb="select"] li:hover {
    background-color: #eff6ff !important;
}

/* Inputs numéricos */
.stNumberInput > div > div > input {
    border-radius: 0.75rem !important;
}

/* Sliders con color institucional */
.stSlider {
    color: var(--primary);
}

/* Separadores suaves */
hr {
    border: none;
    border-top: 1px dashed #e5e7eb;
    margin: 1.2rem 0 1rem 0;
}

/* Dataframe */
.stDataFrame {
    border-radius: 0.75rem;
    overflow: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)

# ==============================
#  Carga del modelo entrenado
# ==============================

@st.cache_resource
def load_pipeline_and_schema():
    here = os.path.dirname(__file__)
    model_path = os.path.join(here, "artefactos", "modelo_atrasos.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

    pipe = joblib.load(model_path)
    prep = pipe.named_steps["prep"]

    num_features = list(prep.transformers_[0][2])
    cat_features = list(prep.transformers_[1][2])
    expected_cols = list(num_features) + list(cat_features)

    return pipe, expected_cols, num_features, cat_features


winner_pipe, EXPECTED_COLS, NUM_FEATS, CAT_FEATS = load_pipeline_and_schema()

LABELS = {0: "FAIL", 1: "PASS"}
BEST_THR = 0.5

SELECTED_FEATURES = [
    "sex", "age", "address", "famsize",
    "Medu", "Fedu",
    "traveltime", "studytime", "failures", "absences",
    "schoolsup", "famsup", "paid", "activities",
    "higher", "internet",
    "famrel", "freetime", "health",
]

VISIBLE_COLS = list(SELECTED_FEATURES)

# ==============================
#  Mapas en español
# ==============================
SEX_OPTS = {"Femenino": "F", "Masculino": "M"}
YESNO_OPTS = {"Sí": "yes", "No": "no"}

ADDRESS_OPTS = {
    "Urbano": "U",
    "Rural": "R",
}

FAMSIZE_OPTS = {
    "≤ 3 miembros": "LE3",
    "> 3 miembros": "GT3",
}

TRAVELTIME_HELP = "1: <15min, 2: 15–30min, 3: 30–60min, 4: >60min"
STUDYTIME_HELP = "1:<2h, 2:2–5h, 3:5–10h, 4:>10h"

# ==============================
#  Helper columnas
# ==============================
def ensure_expected_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in EXPECTED_COLS:
        if col not in df.columns:
            if col in NUM_FEATS:
                df[col] = 0
            else:
                df[col] = ""
    return df[EXPECTED_COLS]

# ==============================
#  Header estilo colegio
# ==============================
here = os.path.dirname(__file__)
logo_path = os.path.join(here, "assets", "logo_union_nana.png")  # ajusta si usas otra ruta

col_logo, col_title = st.columns([1, 4])

with col_logo:
    if os.path.exists(logo_path):
        st.image(logo_path, width=90)
    else:
        st.markdown(
            "<div style='font-size:2.5rem;color:#003c71;'>A</div>",
            unsafe_allow_html=True,
        )

with col_title:
    st.markdown(
        """
<div style="
    background:#003c71;
    border-radius:1rem;
    padding:0.9rem 1.2rem;
    color:white;
    display:flex;
    flex-direction:column;
    gap:0.1rem;
">
  <div style="font-size:1.4rem;font-weight:700;">
    Panel de acompañamiento académico
  </div>
  <div style="font-size:0.95rem;opacity:0.9;">
    Colegio Adventista Unión de Ñaña · Predicción de aprobación (PASS / FAIL)
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

st.caption(
    "Herramienta de apoyo para tutores, psicopedagogía y dirección académica. "
    "No reemplaza la evaluación integral del estudiante."
)

tab_ind, tab_batch = st.tabs(["🧑‍🎓 Predicción individual", "📂 Predicción por lote (CSV)"])

# ==============================
#  Predicción individual
# ==============================
with tab_ind:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    with st.form("form_pass_fail"):
        st.markdown(
            '<h4 style="margin-bottom:0.75rem;">🎯 Predicción individual</h4>',
            unsafe_allow_html=True,
        )

        # --- 1. Datos personales y entorno ---
        st.markdown("#### 1. Datos personales y del entorno 🧍‍♀️🏡")
        c1, c2, c3 = st.columns(3)
        with c1:
            sex_es = st.selectbox("Sexo", list(SEX_OPTS.keys()))
            age = st.number_input("Edad", min_value=10, max_value=25, value=16)
        with c2:
            address_es = st.selectbox("Zona de residencia", list(ADDRESS_OPTS.keys()))
            famsize_es = st.selectbox("Tamaño de familia", list(FAMSIZE_OPTS.keys()))
        with c3:
            traveltime = st.slider("Tiempo de viaje a la escuela", 1, 4, 1, help=TRAVELTIME_HELP)
            studytime = st.slider("Horas de estudio semanal", 1, 4, 2, help=STUDYTIME_HELP)

        st.markdown("---")

        # --- 2. Contexto familiar y escolar ---
        st.markdown("#### 2. Contexto familiar y escolar 🏠📚")
        c4, c5, c6 = st.columns(3)
        with c4:
            Medu = st.slider(
                "Educación de la madre", 0, 4, 2,
                help="0: ninguna, 1: primaria, 2: 5º-9º, 3: secundaria, 4: superior",
            )
            Fedu = st.slider(
                "Educación del padre", 0, 4, 2,
                help="0: ninguna, 1: primaria, 2: 5º-9º, 3: secundaria, 4: superior",
            )
        with c5:
            famrel = st.slider("Relación familiar", 1, 5, 4, help="1 = muy mala, 5 = excelente")
            freetime = st.slider(
                "Tiempo libre después de clases", 1, 5, 3,
                help="1 = muy poco, 5 = mucho",
            )
        with c6:
            absences = st.number_input("Inasistencias totales", min_value=0, max_value=100, value=0)
            failures = st.slider("Repeticiones previas", 0, 4, 0, help="Número de cursos repetidos")

        st.markdown("---")

        # --- 3. Apoyos y hábitos académicos ---
        st.markdown("#### 3. Apoyos y hábitos académicos ✏️💻")
        c7, c8 = st.columns(2)
        with c7:
            schoolsup_es = st.selectbox("Apoyo educativo del colegio", list(YESNO_OPTS.keys()))
            famsup_es = st.selectbox("Apoyo educativo de la familia", list(YESNO_OPTS.keys()))
            paid_es = st.selectbox("Clases particulares pagadas", list(YESNO_OPTS.keys()))
        with c8:
            activities_es = st.selectbox("Actividades extracurriculares", list(YESNO_OPTS.keys()))
            higher_es = st.selectbox("Desea estudios superiores", list(YESNO_OPTS.keys()))
            internet_es = st.selectbox("Acceso a Internet en casa", list(YESNO_OPTS.keys()))

        health = st.slider("Salud general", 1, 5, 4, help="1 = muy mala, 5 = muy buena")

        submitted = st.form_submit_button("Predecir aprobación ✅")

    if submitted:
        sex = SEX_OPTS[sex_es]
        address = ADDRESS_OPTS[address_es]
        famsize = FAMSIZE_OPTS[famsize_es]

        schoolsup = YESNO_OPTS[schoolsup_es]
        famsup = YESNO_OPTS[famsup_es]
        paid = YESNO_OPTS[paid_es]
        activities = YESNO_OPTS[activities_es]
        higher = YESNO_OPTS[higher_es]
        internet = YESNO_OPTS[internet_es]

        data = {
            "sex": sex,
            "age": age,
            "address": address,
            "famsize": famsize,
            "Medu": Medu,
            "Fedu": Fedu,
            "traveltime": traveltime,
            "studytime": studytime,
            "failures": failures,
            "absences": absences,
            "schoolsup": schoolsup,
            "famsup": famsup,
            "paid": paid,
            "activities": activities,
            "higher": higher,
            "internet": internet,
            "famrel": famrel,
            "freetime": freetime,
            "health": health,
        }

        df = pd.DataFrame([data])
        df = ensure_expected_columns(df)

        proba_pass = float(winner_pipe.predict_proba(df)[0, 1])

        pred_int = int(proba_pass >= BEST_THR)
        pred_label = LABELS[pred_int]

        if proba_pass >= BEST_THR:
            explicacion = (
                f"Como {proba_pass:.3f} ≥ {BEST_THR:.2f}, "
                f"el modelo clasifica como **PASS (1)**."
            )
        else:
            explicacion = (
                f"Como {proba_pass:.3f} < {BEST_THR:.2f}, "
                f"el modelo clasifica como **FAIL (0)**."
            )

        colA, colB = st.columns(2)

        with colA:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Probabilidad PASS = 1</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{proba_pass:.3f}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="metric-sub">Equivalente a {proba_pass*100:.1f}% · Umbral: {BEST_THR:.2f}</div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)
            st.progress(min(max(proba_pass, 0.0), 1.0))

        with colB:
            card_class = "metric-card pass" if pred_int == 1 else "metric-card"
            st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Decisión del modelo</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{pred_label} ({pred_int})</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="metric-sub">{explicacion}</div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("#### Distribución de probabilidad 📊")
        prob_df = pd.DataFrame(
            {"Clase": ["FAIL", "PASS"], "Probabilidad": [1 - proba_pass, proba_pass]}
        )
        st.bar_chart(prob_df.set_index("Clase"))

    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
#  Predicción por lote (CSV)
# ==============================
with tab_batch:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        '<h4 style="margin-bottom:0.2rem;">📂 Predicción por lote (CSV)</h4>',
        unsafe_allow_html=True,
    )

    st.write(
        "Sube un archivo **CSV** con, al menos, las siguientes columnas:\n\n"
        "`" + ", ".join(VISIBLE_COLS) + "`\n\n"
        "Cualquier columna faltante que el modelo espere se rellenará con valores neutros."
    )

    file = st.file_uploader("Archivo CSV", type=["csv"])

    if file is not None:
        df_in = pd.read_csv(file)

        faltantes_visibles = [c for c in VISIBLE_COLS if c not in df_in.columns]
        if faltantes_visibles:
            st.warning(
                "Faltan columnas en el CSV (se completarán con valores por defecto):\n\n- "
                + "\n- ".join(faltantes_visibles)
            )

        df_in = ensure_expected_columns(df_in)

        proba = winner_pipe.predict_proba(df_in)[:, 1]
        pred_int = (proba >= BEST_THR).astype(int)
        pred_label = [LABELS[int(z)] for z in pred_int]

        df_out = df_in.copy()
        df_out["proba_pass"] = proba
        df_out["pred_int"] = pred_int
        df_out["pred_label"] = pred_label

        st.write("Vista previa de resultados:")
        st.dataframe(df_out.head())

        st.markdown("#### Distribución de predicciones (FAIL / PASS) 🧮")
        counts = (
            pd.Series(pred_label)
            .value_counts()
            .rename_axis("Clase")
            .reset_index(name="Cantidad")
        )
        st.bar_chart(counts.set_index("Clase"))

        csv_out = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar resultados (CSV)",
            data=csv_out,
            file_name="predicciones_pass_fail.csv",
            mime="text/csv",
        )

    st.markdown('</div>', unsafe_allow_html=True)