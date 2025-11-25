import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
import altair as alt

# ==============================
#  Configuración general
# ==============================
st.set_page_config(
    page_title="Panel de Aprobación – Colegio Unión de Ñaña",
    page_icon="🎓",
    layout="wide",
)

# ==============================
#  Estilos globales (solo diseño)
# ==============================
st.markdown(
    """
<style>
:root {
    --primary: #0857c7;          /* azul colegio */
    --primary-soft: #e5efff;
    --accent: #22c55e;           /* verde acción */
    --accent-soft: #e9fdf2;
    --bg-page: #f5f7fb;
    --bg-card: #ffffff;
    --border-subtle: #e5e7eb;
    --text-main: #111827;
    --text-muted: #6b7280;
}

/* Fondo y contenedor */
main, .stApp {
    background: var(--bg-page);
}
.block-container {
    padding-top: 2.2rem;
    padding-bottom: 2rem;
    max-width: 1200px !important;
}

/* Tipografía */
html, body, .stApp, .stMarkdown, p, li, span, label,
h1, h2, h3, h4, h5, h6, .stCaption {
    color: var(--text-main);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Hero banner */
.hero-banner {
    margin-bottom: 1.6rem;
    background:
        radial-gradient(circle at 0% 0%, #93c5fd 0, transparent 55%),
        radial-gradient(circle at 100% 0%, #fde68a 0, transparent 55%),
        linear-gradient(90deg, #003c71, #2563eb);
    border-radius: 1.25rem;
    padding: 1.1rem 1.5rem;
    color: #f9fafb;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 18px 30px rgba(15, 23, 42, 0.25);
}
.hero-left {
    display: flex;
    gap: 0.9rem;
    align-items: flex-start;
}
.hero-icon {
    width: 52px;
    height: 52px;
    border-radius: 999px;
    display:flex;
    align-items:center;
    justify-content:center;
    background: rgba(15,23,42,0.25);
    font-size: 1.8rem;
}
.hero-title {
    font-size: 1.32rem;
    font-weight: 750;
}
.hero-sub {
    font-size: 0.9rem;
    opacity: 0.9;
}

/* Tabs */
.stTabs {
    margin-top: 0.8rem;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    padding: 0.5rem 1.25rem;
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

/* Tarjetas */
.card {
    background-color: var(--bg-card);
    border-radius: 1.15rem;
    padding: 1.6rem 1.7rem;
    box-shadow: 0 18px 35px rgba(15, 23, 42, 0.07);
    border: 1px solid var(--border-subtle);
    margin-top: 1.2rem;
}
.subcard {
    background-color: #ffffff;
    border-radius: 0.9rem;
    padding: 1.0rem 1.1rem 1.1rem 1.1rem;
    border: 1px solid #e5e7eb;
    margin-top: 0.9rem;
}

/* Secciones */
.section-title {
    font-size: 1.02rem;
    font-weight: 650;
    margin-bottom: 0.4rem;
}
.section-caption {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
}

/* Tarjetas de métricas */
.metric-card {
    border-radius: 0.9rem;
    padding: 0.9rem 1.1rem;
    background: linear-gradient(135deg, var(--primary-soft), #eff4ff);
    color: var(--text-main);
    border: 1px solid #bfdbfe;
}
.metric-card.pass {
    background: linear-gradient(135deg, var(--accent-soft), #fefce8);
    border-color: #bbf7d0;
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
    background: linear-gradient(90deg, var(--accent), #16a34a);
    border: none;
    color: #f9fafb;
    font-weight: 700;
    padding: 0.55rem 1.9rem;
    box-shadow: 0 10px 24px rgba(34, 197, 94, 0.35);
}
.stButton>button:hover {
    filter: brightness(1.06);
}

/* Selects */
.stSelectbox > div > div {
    background-color: #ffffff !important;
    color: var(--text-main) !important;
    border-radius: 0.75rem !important;
    border: 1px solid #d1d5db !important;
}
.stSelectbox svg {
    color: var(--text-muted) !important;
}
div[data-baseweb="select"] ul {
    background-color: #ffffff !important;
}
div[data-baseweb="select"] li {
    color: var(--text-main) !important;
}
div[data-baseweb="select"] li:hover {
    background-color: #eff6ff !important;
}

/* Number input */
.stNumberInput > div {
    background: #ffffff !important;
    border-radius: 0.75rem !important;
    border: 1px solid #d1d5db !important;
}
.stNumberInput input {
    background: #ffffff !important;
    color: var(--text-main) !important;
}
.stNumberInput button {
    background: #eff6ff !important;
    border-radius: 0.75rem !important;
    border: none !important;
}

/* Inputs en general */
input[type="number"], input[type="text"] {
    background-color: #ffffff !important;
    color: var(--text-main) !important;
}

/* Sliders */
.stSlider > div[data-baseweb="slider"] > div {
    color: var(--primary) !important;
}

/* Separador suave */
hr {
    border: none;
    border-top: 1px dashed #e5e7eb;
    margin: 1.1rem 0 1.0rem 0;
}

/* DataFrame claro */
.stDataFrame, .stDataFrame [data-testid="stTable"] {
    background-color: #ffffff !important;
    color: var(--text-main) !important;
}
.stDataFrame tbody tr:nth-child(even) {
    background-color: #f9fafb !important;
}

/* Gráficos Altair */
.stAltairChart {
    background-color: #ffffff;
    border-radius: 0.9rem;
    padding: 0.75rem;
    border: 1px solid #e5e7eb;
}
.vega-embed, .vega-embed canvas {
    background-color: #ffffff !important;
}

/* Tooltips (icono de ayuda) */
div[data-testid="stTooltipContent"] {
    background-color: #ffffff !important;
    color: var(--text-main) !important;
    border-radius: 0.6rem !important;
    border: 1px solid #e5e7eb !important;
    box-shadow: 0 12px 30px rgba(15,23,42,0.18) !important;
    font-size: 0.8rem !important;
}

/* Notas pequeñas */
.small-note {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 0.4rem;
}
.info-chip {
    display:inline-flex;
    align-items:center;
    gap:0.35rem;
    padding:0.12rem 0.55rem;
    border-radius:999px;
    background:#e5efff;
    color:#1d4ed8;
    font-size:0.75rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ==============================
#  Carga del modelo (LOGIC NO TOUCH)
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

ADDRESS_OPTS = {"Urbano": "U", "Rural": "R"}
FAMSIZE_OPTS = {"≤ 3 miembros": "LE3", "> 3 miembros": "GT3"}

TRAVELTIME_HELP = "1: <15min, 2: 15–30min, 3: 30–60min, 4: >60min"
STUDYTIME_HELP = "1:<2h, 2:2–5h, 3:5–10h, 4:>10h"

DEFAULT_TRAVELTIME = 1  # tiempo de viaje fijo (<15 min)

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
#  HEADER
# ==============================
here = os.path.dirname(__file__)
logo_path = os.path.join(here, "assets", "logo_union_nana.png")

hero_cols = st.columns([1, 4])
with hero_cols[0]:
    if os.path.exists(logo_path):
        st.image(logo_path, width=80)
    else:
        st.markdown(
            "<div style='font-size:2.4rem;color:#ffffff;"
            "background:#003c71;border-radius:999px;width:64px;height:64px;"
            "display:flex;align-items:center;justify-content:center;'>A</div>",
            unsafe_allow_html=True,
        )

with hero_cols[1]:
    st.markdown(
        """
<div class="hero-banner">
  <div class="hero-left">
    <div class="hero-icon">🎓</div>
    <div>
      <div class="hero-title">Panel de acompañamiento académico</div>
      <div class="hero-sub">
        Colegio Adventista Unión de Ñaña · Predicción de aprobación (PASS / FAIL) a partir de hábitos y contexto del estudiante.
      </div>
    </div>
  </div>
  <div style="font-size:0.8rem; text-align:right; opacity:0.9;">
    Versión institucional<br/>
    <span style="font-weight:600;">Uso orientativo para tutores y psicopedagogía</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

st.caption(
    "Esta herramienta **no reemplaza** la evaluación integral del estudiante; "
    "sirve como apoyo para decisiones pedagógicas."
)

tab_ind, tab_batch = st.tabs(["🧑‍🎓 Predicción individual", "📂 Predicción por lote (CSV)"])

# ==============================
#  PREDICCIÓN INDIVIDUAL
# ==============================
with tab_ind:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Layout nuevo: columna izquierda = formulario, derecha = resumen / guía
    col_form, col_side = st.columns([3, 2])

    with col_form:
        with st.form("form_pass_fail"):

            # ---- Bloque 1: Ficha básica del estudiante ----
            st.markdown(
                '<div class="section-title">1. Ficha del estudiante 🧍‍♀️</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-caption">Completa los datos generales del estudiante.</div>',
                unsafe_allow_html=True,
            )
            sf1, sf2 = st.columns(2)
            with sf1:
                sex_es = st.selectbox("Sexo", list(SEX_OPTS.keys()))
                age = st.number_input("Edad", min_value=10, max_value=25, value=16)
            with sf2:
                address_es = st.selectbox("Zona de residencia", list(ADDRESS_OPTS.keys()))
                famsize_es = st.selectbox("Tamaño de familia", list(FAMSIZE_OPTS.keys()))

            st.markdown("<hr/>", unsafe_allow_html=True)

            # ---- Bloque 2: Entorno familiar y bienestar ----
            st.markdown(
                '<div class="section-title">2. Entorno familiar y bienestar 🏠💬</div>',
                unsafe_allow_html=True,
            )

            with st.container():
                c_ed, c_rel = st.columns(2)
                with c_ed:
                    Medu = st.slider(
                        "Educación de la madre",
                        0, 4, 2,
                        help="0: ninguna, 1: primaria, 2: 5º-9º, 3: secundaria, 4: superior",
                    )
                    Fedu = st.slider(
                        "Educación del padre",
                        0, 4, 2,
                        help="0: ninguna, 1: primaria, 2: 5º-9º, 3: secundaria, 4: superior",
                    )
                with c_rel:
                    famrel = st.slider(
                        "Relación familiar",
                        1, 5, 4,
                        help="1 = muy mala, 5 = excelente",
                    )
                    freetime = st.slider(
                        "Tiempo libre después de clases",
                        1, 5, 3,
                        help="1 = muy poco, 5 = mucho",
                    )

                c_rep, c_abs = st.columns(2)
                with c_rep:
                    failures = st.slider(
                        "Repeticiones previas",
                        0, 4, 0,
                        help="Número de cursos repetidos hasta la fecha.",
                    )
                with c_abs:
                    absences = st.number_input(
                        "Inasistencias totales",
                        min_value=0,
                        max_value=100,
                        value=0,
                        help="Cantidad total de inasistencias acumuladas en el año.",
                    )

                health = st.slider(
                    "Salud general",
                    1, 5, 4,
                    help="Percepción general de salud: 1 = muy mala, 5 = muy buena.",
                )

            st.markdown("<hr/>", unsafe_allow_html=True)

            # ---- Bloque 3: Hábitos y apoyos académicos ----
            st.markdown(
                '<div class="section-title">3. Hábitos y apoyos académicos ✏️📚</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="section-caption">Registra las horas de estudio y los apoyos educativos con los que cuenta.</div>',
                unsafe_allow_html=True,
            )

            ch1, ch2 = st.columns(2)
            with ch1:
                studytime = st.slider(
                    "Horas de estudio semanal",
                    1, 4, 2,
                    help="1:<2h, 2:2–5h, 3:5–10h, 4:>10h de estudio fuera de clases.",
                )
                schoolsup_es = st.selectbox(
                    "Apoyo educativo del colegio",
                    list(YESNO_OPTS.keys()),
                    help="Apoyos adicionales brindados por el colegio (talleres, refuerzo, etc.)",
                )
                famsup_es = st.selectbox(
                    "Apoyo educativo de la familia",
                    list(YESNO_OPTS.keys()),
                    help="Apoyo en tareas, acompañamiento y seguimiento desde el hogar.",
                )
            with ch2:
                paid_es = st.selectbox(
                    "Clases particulares pagadas",
                    list(YESNO_OPTS.keys()),
                    help="Refuerzos externos como academias o profesores particulares.",
                )
                activities_es = st.selectbox(
                    "Actividades extracurriculares",
                    list(YESNO_OPTS.keys()),
                    help="Participación en deportes, arte, música u otras actividades formativas.",
                )
                higher_es = st.selectbox(
                    "Desea estudios superiores",
                    list(YESNO_OPTS.keys()),
                    help="Intención declarada de continuar estudios técnicos o universitarios.",
                )
                internet_es = st.selectbox(
                    "Acceso a Internet en casa",
                    list(YESNO_OPTS.keys()),
                    help="Disponibilidad de conexión estable para realizar tareas y trabajos.",
                )

            st.markdown(
                '<div class="small-note">Al hacer clic en <b>"Predecir aprobación"</b> se calculará la probabilidad de que el estudiante apruebe el año escolar.</div>',
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Predecir aprobación ✅")

    # ---- Columna derecha: guía y contexto ----
    with col_side:
        st.markdown(
            """
<div class="subcard">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.3rem;">
    <div style="font-weight:650;">Guía de lectura rápida</div>
    <div class="info-chip">ℹ️ Solo apoyo orientativo</div>
  </div>
  <p style="font-size:0.85rem;color:#4b5563;margin-bottom:0.3rem;">
    • Valores de probabilidad cercanos a <b>1</b> indican una alta probabilidad de <b>aprobación (PASS)</b>.<br/>
    • Valores cercanos a <b>0</b> sugieren riesgo de <b>desaprobación (FAIL)</b> y requieren seguimiento más profundo.<br/>
    • Usa siempre esta información junto con entrevistas, observaciones en aula y reportes de los docentes.
  </p>
  <p style="font-size:0.85rem;color:#4b5563;margin-top:0.5rem;margin-bottom:0;">
    Te sugerimos revisar especialmente:
  </p>
  <ul style="font-size:0.84rem;color:#4b5563;margin-top:0.3rem;padding-left:1.1rem;">
    <li>Estudiantes con muchas <b>inasistencias</b> o varias <b>repeticiones previas</b>.</li>
    <li>Niveles muy bajos de <b>apoyo familiar</b> y <b>horas de estudio</b>.</li>
    <li>Diferencias importantes entre la percepción de <b>salud</b> y el rendimiento observado.</li>
  </ul>
</div>
""",
            unsafe_allow_html=True,
        )

    # ----------------- POST-PREDICCIÓN -----------------
    if submitted:
        # Mapear selecciones a códigos originales (LÓGICA IGUAL)
        sex = SEX_OPTS[sex_es]
        address = ADDRESS_OPTS[address_es]
        famsize = FAMSIZE_OPTS[famsize_es]

        schoolsup = YESNO_OPTS[schoolsup_es]
        famsup = YESNO_OPTS[famsup_es]
        paid = YESNO_OPTS[paid_es]
        activities = YESNO_OPTS[activities_es]
        higher = YESNO_OPTS[higher_es]
        internet = YESNO_OPTS[internet_es]

        traveltime = DEFAULT_TRAVELTIME  # fijo, ya no se pide en UI

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

        st.markdown("<hr/>", unsafe_allow_html=True)

        # Métricas en dos columnas
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

        # Gráfico de probabilidad
        st.markdown("#### Distribución de probabilidad 📊")
        prob_df = pd.DataFrame(
            {"Clase": ["FAIL", "PASS"], "Probabilidad": [1 - proba_pass, proba_pass]}
        )

        chart = (
            alt.Chart(prob_df)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(
                x=alt.X("Clase:N", sort=["FAIL", "PASS"], title=None),
                y=alt.Y(
                    "Probabilidad:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%", title="Probabilidad"),
                ),
                color=alt.Color(
                    "Clase:N",
                    scale=alt.Scale(range=["#fecaca", "#bfdbfe"]),
                    legend=None,
                ),
            )
            .properties(height=260)
            .configure_view(strokeWidth=0, fill="#ffffff")
        )

        st.altair_chart(chart, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ==============================
#  PREDICCIÓN POR LOTE (CSV)
# ==============================
with tab_batch:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### 📂 Predicción por lote (CSV)")

    st.write(
        "Sube un archivo **CSV** con, al menos, las siguientes columnas:\n\n"
        "`" + ", ".join(VISIBLE_COLS) + "`\n\n"
        "Las columnas faltantes que el modelo espere se rellenarán con valores neutros."
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

        chart2 = (
            alt.Chart(counts)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(
                x=alt.X("Clase:N", sort=["FAIL", "PASS"], title=None),
                y=alt.Y("Cantidad:Q", axis=alt.Axis(title="Número de estudiantes")),
                color=alt.Color(
                    "Clase:N",
                    scale=alt.Scale(range=["#fecaca", "#bfdbfe"]),
                    legend=None,
                ),
            )
            .properties(height=260)
            .configure_view(strokeWidth=0, fill="#ffffff")
        )

        st.altair_chart(chart2, use_container_width=True)

        csv_out = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar resultados (CSV)",
            data=csv_out,
            file_name="predicciones_pass_fail.csv",
            mime="text/csv",
        )

    st.markdown("</div>", unsafe_allow_html=True)
