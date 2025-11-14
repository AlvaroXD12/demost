import streamlit as st
import joblib
import os
import pandas as pd

# ==============================
#  Configuración general
# ==============================
st.set_page_config(
    page_title="Clasificación — Atraso escolar",
    page_icon="🎓",
    layout="centered",
)

# Estilos (fondo CLARO + tarjetas + texto negro y selects oscuros)
st.markdown(
    """
<style>
/* Fondo general claro */
main, .stApp {
    background: #f3f4f6;
}

/* Texto general negro (títulos, párrafos, labels, caption) */
html, body, .stApp, .stMarkdown, p, li, span, label, h1, h2, h3, h4, h5, h6, .stCaption {
    color: #111827 !important;
}

/* Tabs tipo pastilla */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.15rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    padding: 0.35rem 0.9rem;
    background-color: #e5e7eb;
    color: #374151;
    font-weight: 500;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: #2563eb !important;
    color: #f9fafb !important;
}

/* Tarjeta principal (contenedor grande) */
.card {
    background-color: #ffffff;
    border-radius: 1rem;
    padding: 1.5rem 1.75rem;
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.12);
    border: 1px solid rgba(37, 99, 235, 0.25);  /* borde azul suave */
    color: #111827;
}

/* Tarjetas de métricas */
.metric-card {
    border-radius: 0.9rem;
    padding: 0.9rem 1.1rem;
    background: #111827;
    color: #f9fafb;
    border: 1px solid #1f2937;
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
    color: #d1d5db;
}

/* Botón principal */
.stButton>button {
    border-radius: 999px;
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    border: none;
    color: #f9fafb;
    font-weight: 600;
    padding: 0.45rem 1.5rem;
}
.stButton>button:hover {
    filter: brightness(1.05);
}

/* ====== SELECTS: fondo oscuro + texto blanco ====== */

/* Caja del valor seleccionado */
.stSelectbox > div > div {
    background-color: #111827 !important;
    color: #f9fafb !important;
    border-radius: 0.75rem !important;
}

/* Icono del select (flecha) */
.stSelectbox svg {
    color: #f9fafb !important;
}

/* Menú desplegable */
div[data-baseweb="select"] ul {
    background-color: #111827 !important;
}
div[data-baseweb="select"] li {
    color: #f9fafb !important;
}
div[data-baseweb="select"] li:hover {
    background-color: #1f2937 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ==============================
#  Carga del modelo entrenado
# ==============================
ART_DIR = "artefactos"   # carpeta donde subiste el .joblib

@st.cache_resource
def load_pipeline():
    model_path = os.path.join(ART_DIR, "modelo_atrasos.joblib")
    return joblib.load(model_path)

winner_pipe = load_pipeline()

# Mapa de etiquetas
LABEL_MAP = {"NO_ATRASO": 0, "ATRASO": 1}
REV_LABEL = {v: k for k, v in LABEL_MAP.items()}

BEST_THR = 0.5

# Variables usadas por el modelo
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
#  Header
# ==============================
st.markdown(
    '<h3 style="font-weight:700; margin-bottom:0.15rem;">🤖 Clasificación — Atraso escolar</h3>',
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
    
    with st.form("form_atraso"):
        st.markdown(
            '<h4 style="margin-bottom:0.2rem;">Predicción individual</h4>',
            unsafe_allow_html=True,
        )
        st.caption("Completa los datos del estudiante y presiona **Predecir atraso**.")

        # 3 columnas, cada una con 6 controles
        col1, col2, col3 = st.columns(3)

        # ============================
        # Columna 1 (3 cajas + 3 sliders)
        # ============================
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

        # ============================
        # Columna 2 (3 cajas + 3 sliders)
        # ============================
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

        # ============================
        # Columna 3 (2 cajas + 4 sliders)
        # ============================
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
        # Mapear selecciones en español a códigos originales
        sex = SEX_OPTS[sex_es]
        schoolsup = YESNO_OPTS[schoolsup_es]
        famsup = YESNO_OPTS[famsup_es]
        activities = YESNO_OPTS[activities_es]
        higher = YESNO_OPTS[higher_es]
        internet = YESNO_OPTS[internet_es]

        # Construir diccionario con las features que usa el modelo
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

        # Probabilidad de ATRASO (clase 1)
        proba_atraso = float(winner_pipe.predict_proba(df)[0, 1])
        pred_int = int(proba_atraso >= BEST_THR)
        pred_label = REV_LABEL[pred_int]

        # Métricas tipo “tarjeta”
        colA, colB = st.columns(2)
        with colA:
            
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
            
            st.markdown('<div class="metric-label">Decisión del modelo</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{pred_label} ({pred_int})</div>',
                unsafe_allow_html=True,
            )
            if pred_int == 1:
                st.markdown(
                    '<div class="metric-sub">El estudiante está <b>en riesgo de atraso</b>.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="metric-sub">El estudiante <b>no</b> está en riesgo de atraso.</div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
#  Predicción por lote (CSV)
# ==============================
with tab_batch:
    
    st.markdown(
        '<h4 style="margin-bottom:0.2rem;">Predicción por lote (CSV)</h4>',
        unsafe_allow_html=True,
    )

    st.write(
        "Sube un archivo **CSV** que contenga al menos las siguientes columnas "
        f"(con estos nombres exactos): `{', '.join(FEATURES)}`."
    )

    file = st.file_uploader("Archivo CSV", type=["csv"])

    if file is not None:
        df_in = pd.read_csv(file)

        faltantes = [c for c in FEATURES if c not in df_in.columns]
        if faltantes:
            st.error(
                "Faltan columnas en el CSV:\n\n- " + "\n- ".join(faltantes)
                + "\n\nAsegúrate de que los nombres coincidan exactamente."
            )
        else:
            proba = winner_pipe.predict_proba(df_in[FEATURES])[:, 1]
            pred_int = (proba >= BEST_THR).astype(int)
            pred_label = [REV_LABEL[i] for i in pred_int]

            df_out = df_in.copy()
            df_out["proba_atraso"] = proba
            df_out["pred_int"] = pred_int
            df_out["pred_label"] = pred_label

            st.write("Vista previa de resultados:")
            st.dataframe(df_out.head())

            csv_out = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Descargar resultados (CSV)",
                data=csv_out,
                file_name="predicciones_atraso.csv",
                mime="text/csv",
            )

    st.markdown('</div>', unsafe_allow_html=True)
