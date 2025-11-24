import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np

# ==============================
#  Configuración general (MODO OSCURO)
# ==============================
st.set_page_config(
    page_title="Clasificación — Atraso escolar",
    page_icon="🎓",
    layout="centered",
)

st.markdown(
    """
<style>
/* Fondo general oscuro */
main, .stApp {
    background: #020617;
}

/* Texto general claro */
html, body, .stApp, .stMarkdown, p, li, span, label,
h1, h2, h3, h4, h5, h6, .stCaption {
    color: #e5e7eb !important;
}

/* Tabs tipo pastilla */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.15rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    padding: 0.35rem 0.9rem;
    background-color: #0f172a;
    color: #cbd5f5;
    font-weight: 500;
    border: 1px solid #1e293b;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #2563eb, #1d4ed8) !important;
    color: #f9fafb !important;
}

/* Tarjeta principal */
.card {
    background-color: #020617;
    border-radius: 1rem;
    padding: 1.5rem 1.75rem;
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.9);
    border: 1px solid rgba(148, 163, 184, 0.6);
    color: #e5e7eb;
}

/* Tarjetas de métricas */
.metric-card {
    border-radius: 0.9rem;
    padding: 0.9rem 1.1rem;
    background: radial-gradient(circle at top left, #1e293b, #020617);
    color: #f9fafb;
    border: 1px solid #334155;
}
.metric-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: #9ca3af;
}
.metric-value {
    font-size: 1.7rem;
    font-weight: 700;
}
.metric-sub {
    font-size: 0.9rem;
    color: #cbd5f5;
}

/* Botón principal */
.stButton>button {
    border-radius: 999px;
    background: linear-gradient(90deg, #22c55e, #16a34a);
    border: none;
    color: #0f172a;
    font-weight: 700;
    padding: 0.5rem 1.7rem;
}
.stButton>button:hover {
    filter: brightness(1.1);
}

/* Selects oscuros */
.stSelectbox > div > div {
    background-color: #020617 !important;
    color: #e5e7eb !important;
    border-radius: 0.75rem !important;
    border: 1px solid #475569 !important;
}
.stSelectbox svg {
    color: #e5e7eb !important;
}
div[data-baseweb="select"] ul {
    background-color: #020617 !important;
}
div[data-baseweb="select"] li {
    color: #e5e7eb !important;
}
div[data-baseweb="select"] li:hover {
    background-color: #111827 !important;
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
    # ruta absoluta a artefactos/modelo_atrasos.joblib
    here = os.path.dirname(__file__)
    model_path = os.path.join(here, "artefactos", "modelo_atrasos.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

    pipe = joblib.load(model_path)

    # El primer paso del pipeline es "prep" (ColumnTransformer con num/cat)
    prep = pipe.named_steps["prep"]

    # Columnas numéricas y categóricas usadas en el entrenamiento
    num_features = list(prep.transformers_[0][2])
    cat_features = list(prep.transformers_[1][2])
    expected_cols = list(num_features) + list(cat_features)

    return pipe, expected_cols, num_features, cat_features


winner_pipe, EXPECTED_COLS, NUM_FEATS, CAT_FEATS = load_pipeline_and_schema()

# Mapa de etiquetas solo para mostrar
LABEL_MAP = {"NO_ATRASO": 0, "ATRASO": 1}
REV_LABEL = {v: k for k, v in LABEL_MAP.items()}

BEST_THR = 0.5

# Variables visibles para el usuario
FEATURES = [
    "sex", "age", "studytime", "failures", "absences",
    "schoolsup", "famsup", "activities", "higher",
    "internet", "goout", "Dalc", "Walc",
    "famrel", "freetime", "health", "Medu", "Fedu",
]

# ==============================
#  Mapas para mostrar en español
# ==============================
SEX_OPTS = {
    "Femenino": "F",
    "Masculino": "M",
}
YESNO_OPTS = {
    "Sí": "yes",
    "No": "no",
}

# ==============================
#  Helper: asegurar columnas
# ==============================
def ensure_expected_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura que el DataFrame tenga TODAS las columnas usadas en el entrenamiento.
    Las que falten se rellenan con valores neutros.
    """
    for col in EXPECTED_COLS:
        if col not in df.columns:
            if col in NUM_FEATS:
                df[col] = 0
            else:
                df[col] = ""  # categóricas → unknown (OneHotEncoder las ignora)
    return df[EXPECTED_COLS]


# ==============================
#  Header
# ==============================
st.markdown(
    '<h3 style="font-weight:700; margin-bottom:0.15rem;">🎓 Clasificación — Atraso escolar</h3>',
    unsafe_allow_html=True,
)
st.caption(
    "App de inferencia ML para predecir **ATRASO (1)** vs **NO_ATRASO (0)** "
    "a partir de hábitos y contexto del estudiante."
)

tab_ind, tab_batch = st.tabs(["🔹 Predicción individual", "📂 Predicción por lote (CSV)"])

# ==============================
#  Predicción individual
# ==============================
with tab_ind:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    with st.form("form_atraso"):
        st.markdown(
            '<h4 style="margin-bottom:0.2rem;">Predicción individual</h4>',
            unsafe_allow_html=True,
        )
        st.caption("Completa los datos del estudiante y presiona **Predecir atraso**.")

        col1, col2, col3 = st.columns(3)

        # Columna 1
        with col1:
            sex_es = st.selectbox("Sexo", list(SEX_OPTS.keys()))
            schoolsup_es = st.selectbox(
                "Apoyo educativo del colegio",
                list(YESNO_OPTS.keys()),
            )
            age = st.number_input("Edad", min_value=15, max_value=25, value=17)

            health = st.slider(
                "Salud actual",
                1, 5, 4,
                help="1 = muy mala, 5 = muy buena",
            )
            Dalc = st.slider(
                "Alcohol (días de semana)",
                1, 5, 1,
                help="1 = muy bajo, 5 = muy alto",
            )
            Walc = st.slider(
                "Alcohol (fin de semana)",
                1, 5, 1,
                help="1 = muy bajo, 5 = muy alto",
            )

        # Columna 2
        with col2:
            famsup_es = st.selectbox(
                "Apoyo educativo de la familia",
                list(YESNO_OPTS.keys()),
            )
            activities_es = st.selectbox(
                "Actividades extracurriculares",
                list(YESNO_OPTS.keys()),
            )
            absences = st.number_input(
                "Inasistencias",
                min_value=0, max_value=100, value=0,
            )

            studytime = st.slider(
                "Horas de estudio semanal",
                1, 4, 2,
                help="1:<2h, 2:2-5h, 3:5-10h, 4:>10h",
            )
            failures = st.slider(
                "Repeticiones previas",
                0, 4, 0,
                help="Número de veces que repitió curso/asignatura",
            )
            goout = st.slider(
                "Salir con amigos",
                1, 5, 2,
                help="1 = casi nunca, 5 = muy frecuente",
            )

        # Columna 3
        with col3:
            higher_es = st.selectbox(
                "Desea estudios superiores",
                list(YESNO_OPTS.keys()),
            )
            internet_es = st.selectbox(
                "Acceso a Internet en casa",
                list(YESNO_OPTS.keys()),
            )

            famrel = st.slider(
                "Relación con la familia",
                1, 5, 4,
                help="1 = muy mala, 5 = excelente",
            )
            freetime = st.slider(
                "Tiempo libre después de clases",
                1, 5, 3,
                help="1 = muy poco, 5 = mucho",
            )
            Medu = st.slider(
                "Educación de la madre",
                0, 4, 2,
                help="0 = ninguna, 1 = primaria, 2 = 5º-9º, 3 = secundaria, 4 = superior",
            )
            Fedu = st.slider(
                "Educación del padre",
                0, 4, 2,
                help="0 = ninguna, 1 = primaria, 2 = 5º-9º, 3 = secundaria, 4 = superior",
            )

        submitted = st.form_submit_button("Predecir atraso")

    if submitted:
        sex = SEX_OPTS[sex_es]
        schoolsup = YESNO_OPTS[schoolsup_es]
        famsup = YESNO_OPTS[famsup_es]
        activities = YESNO_OPTS[activities_es]
        higher = YESNO_OPTS[higher_es]
        internet = YESNO_OPTS[internet_es]

        data = {
            "sex": sex,
            "age": age,
            "studytime": studytime,
            "failures": failures,
            "absences": absences,
            "schoolsup": schoolsup,
            "famsup": famsup,
            "activities": activities,
            "higher": higher,
            "internet": internet,
            "goout": goout,
            "Dalc": Dalc,
            "Walc": Walc,
            "famrel": famrel,
            "freetime": freetime,
            "health": health,
            "Medu": Medu,
            "Fedu": Fedu,
        }

        df = pd.DataFrame([data])

        # Completar columnas que el modelo espera
        df = ensure_expected_columns(df)

        # Probabilidad de ATRASO (clase 1)
        proba_atraso = float(winner_pipe.predict_proba(df)[0, 1])

        pred_int = int(proba_atraso >= BEST_THR)
        pred_label = "ATRASO" if pred_int == 1 else "NO_ATRASO"

        if proba_atraso >= BEST_THR:
            explicacion = (
                f"Como {proba_atraso:.3f} ≥ {BEST_THR:.2f}, "
                f"el modelo clasifica como **ATRASO (1)**."
            )
        else:
            explicacion = (
                f"Como {proba_atraso:.3f} < {BEST_THR:.2f}, "
                f"el modelo clasifica como **NO_ATRASO (0)**."
            )

        colA, colB = st.columns(2)

        with colA:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Probabilidad ATRASO = 1</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{proba_atraso:.3f}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="metric-sub">Equivalente a {proba_atraso*100:.1f}% &nbsp;|&nbsp; Umbral: {BEST_THR:.2f}</div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)
            st.progress(min(max(proba_atraso, 0.0), 1.0))

        with colB:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
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

        # Gráfico intuitivo NO_ATRASO vs ATRASO
        st.markdown("#### Distribución de probabilidad")
        prob_df = pd.DataFrame(
            {"Clase": ["NO_ATRASO", "ATRASO"], "Probabilidad": [1 - proba_atraso, proba_atraso]}
        )
        st.bar_chart(prob_df.set_index("Clase"))

    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
#  Predicción por lote (CSV)
# ==============================
with tab_batch:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        '<h4 style="margin-bottom:0.2rem;">Predicción por lote (CSV)</h4>',
        unsafe_allow_html=True,
    )

    st.write(
        "Sube un archivo **CSV** que contenga, al menos, las columnas "
        f"`{', '.join(FEATURES)}`. El resto de columnas que el modelo usa se rellenan con valores por defecto."
    )

    file = st.file_uploader("Archivo CSV", type=["csv"])

    if file is not None:
        df_in = pd.read_csv(file)

        faltantes_visibles = [c for c in FEATURES if c not in df_in.columns]
        if faltantes_visibles:
            st.warning(
                "Faltan columnas recomendadas en el CSV:\n\n- "
                + "\n- ".join(faltantes_visibles)
                + "\n\nSe rellenarán las que falten con valores neutros."
            )

        # Aseguramos TODAS las columnas esperadas por el modelo
        df_in = ensure_expected_columns(df_in)

        proba = winner_pipe.predict_proba(df_in)[:, 1]
        pred_int = (proba >= BEST_THR).astype(int)
        pred_label = np.where(pred_int == 1, "ATRASO", "NO_ATRASO")

        df_out = df_in.copy()
        df_out["proba_atraso"] = proba
        df_out["pred_int"] = pred_int
        df_out["pred_label"] = pred_label

        st.write("Vista previa de resultados:")
        st.dataframe(df_out.head())

        # Gráfica de distribución de predicciones
        st.markdown("#### Distribución de predicciones (NO_ATRASO / ATRASO)")
        counts = pd.Series(pred_label).value_counts().rename_axis("Clase").reset_index(name="Cantidad")
        st.bar_chart(counts.set_index("Clase"))

        csv_out = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar resultados (CSV)",
            data=csv_out,
            file_name="predicciones_atraso.csv",
            mime="text/csv",
        )

    st.markdown('</div>', unsafe_allow_html=True)
