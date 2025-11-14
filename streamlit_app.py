import streamlit as st
import joblib
import os
import pandas as pd

# ==============================
#  Carga del modelo entrenado
# ==============================
ART_DIR = "artefactos"   # carpeta donde subiste el .joblib

@st.cache_resource
def load_pipeline():
    # 🔴 Cambia el nombre si tu archivo se llama distinto
    model_path = os.path.join(ART_DIR, "modelo_atraso.joblib")
    return joblib.load(model_path)

winner_pipe = load_pipeline()

# Mapa de etiquetas fijo para tu problema
LABEL_MAP = {"NO_ATRASO": 0, "ATRASO": 1}
REV_LABEL = {v: k for k, v in LABEL_MAP.items()}  # {0:"NO_ATRASO",1:"ATRASO"}

BEST_THR = 0.5  # mismo umbral que usaste en tu notebook

# ==============================
#  Mapas para mostrar en español
# ==============================
SCHOOL_OPTS = {
    "Gabriel Pereira (GP)": "GP",
    "Mousinho da Silveira (MS)": "MS",
}

SEX_OPTS = {
    "Femenino": "F",
    "Masculino": "M",
}

ADDRESS_OPTS = {
    "Urbano": "U",
    "Rural": "R",
}

FAMSIZE_OPTS = {
    "≤ 3 miembros": "LE3",
    "> 3 miembros": "GT3",
}

PSTATUS_OPTS = {
    "Padres juntos": "T",
    "Padres separados": "A",
}

MJOB_OPTS = {
    "Docente": "teacher",
    "Salud": "health",
    "Servicios públicos": "services",
    "Ama de casa": "at_home",
    "Otro": "other",
}

FJOB_OPTS = MJOB_OPTS  # mismo catálogo

REASON_OPTS = {
    "Cerca de casa": "home",
    "Reputación del colegio": "reputation",
    "Preferencia por el curso": "course",
    "Otro motivo": "other",
}

GUARDIAN_OPTS = {
    "Madre": "mother",
    "Padre": "father",
    "Otro": "other",
}

YESNO_OPTS = {
    "Sí": "yes",
    "No": "no",
}

ROMANTIC_OPTS = YESNO_OPTS
INTERNET_OPTS = YESNO_OPTS
HIGHER_OPTS = YESNO_OPTS
NURSERY_OPTS = YESNO_OPTS
ACTIVITIES_OPTS = YESNO_OPTS
SCHOOLSUP_OPTS = YESNO_OPTS
FAMSUP_OPTS = YESNO_OPTS
PAID_OPTS = YESNO_OPTS

# ==============================
#  Interfaz de la aplicación
# ==============================
st.title("Predicción de atraso escolar por hábitos")

st.write(
    """
Esta aplicación usa un modelo de **minería de datos** para predecir si un estudiante
se encuentra en **riesgo de atraso escolar (ATRASO = 1)** en función de sus hábitos,
características familiares y contexto.
"""
)

with st.form("form_atraso"):
    st.subheader("Datos del estudiante")

    col1, col2, col3 = st.columns(3)

    # -------- Columna 1 --------
    with col1:
        school_es = st.selectbox(
            "Colegio",
            list(SCHOOL_OPTS.keys()),
            help="Selecciona el colegio del estudiante"
        )
        sex_es = st.selectbox(
            "Sexo",
            list(SEX_OPTS.keys()),
            help="Sexo biológico del estudiante"
        )
        age = st.number_input(
            "Edad",
            min_value=15,
            max_value=25,
            value=17
        )
        address_es = st.selectbox(
            "Tipo de domicilio",
            list(ADDRESS_OPTS.keys()),
            help="Urbano o rural"
        )
        famsize_es = st.selectbox(
            "Tamaño de la familia",
            list(FAMSIZE_OPTS.keys()),
            help="Número de miembros del hogar"
        )
        Pstatus_es = st.selectbox(
            "Situación de los padres",
            list(PSTATUS_OPTS.keys()),
            help="Si viven juntos o separados"
        )

    # -------- Columna 2 --------
    with col2:
        Medu = st.slider(
            "Educación de la madre",
            0, 4, 2,
            help="0 = ninguna, 1 = primaria, 2 = 5º-9º, 3 = secundaria, 4 = superior"
        )
        Fedu = st.slider(
            "Educación del padre",
            0, 4, 2,
            help="0 = ninguna, 1 = primaria, 2 = 5º-9º, 3 = secundaria, 4 = superior"
        )
        Mjob_es = st.selectbox(
            "Trabajo de la madre",
            list(MJOB_OPTS.keys())
        )
        Fjob_es = st.selectbox(
            "Trabajo del padre",
            list(FJOB_OPTS.keys())
        )
        reason_es = st.selectbox(
            "Razón para elegir el colegio",
            list(REASON_OPTS.keys())
        )
        guardian_es = st.selectbox(
            "Apoderado principal",
            list(GUARDIAN_OPTS.keys())
        )

    # -------- Columna 3 --------
    with col3:
        traveltime = st.slider(
            "Tiempo de viaje al colegio",
            1, 4, 1,
            help="1:<15m, 2:15-30m, 3:30-60m, 4:>1h"
        )
        studytime = st.slider(
            "Horas de estudio semanal",
            1, 4, 2,
            help="1:<2h, 2:2-5h, 3:5-10h, 4:>10h"
        )
        failures = st.slider(
            "Número de repeticiones previas",
            0, 4, 0,
            help="Número de veces que repitió curso/asignatura"
        )
        schoolsup_es = st.selectbox(
            "Apoyo educativo extra del colegio",
            list(SCHOOLSUP_OPTS.keys())
        )
        famsup_es = st.selectbox(
            "Apoyo educativo de la familia",
            list(FAMSUP_OPTS.keys())
        )
        paid_es = st.selectbox(
            "Clases particulares pagadas",
            list(PAID_OPTS.keys())
        )

    col4, col5, col6 = st.columns(3)

    # -------- Columna 4 --------
    with col4:
        activities_es = st.selectbox(
            "Actividades extracurriculares",
            list(ACTIVITIES_OPTS.keys())
        )
        nursery_es = st.selectbox(
            "Asistió a educación inicial",
            list(NURSERY_OPTS.keys())
        )
        higher_es = st.selectbox(
            "Desea estudios superiores",
            list(HIGHER_OPTS.keys())
        )
        internet_es = st.selectbox(
            "Acceso a Internet en casa",
            list(INTERNET_OPTS.keys())
        )
        romantic_es = st.selectbox(
            "Tiene relación romántica",
            list(ROMANTIC_OPTS.keys())
        )

    # -------- Columna 5 --------
    with col5:
        famrel = st.slider(
            "Relación con la familia",
            1, 5, 4,
            help="1 = muy mala, 5 = excelente"
        )
        freetime = st.slider(
            "Tiempo libre después de clase",
            1, 5, 3,
            help="1 = muy poco, 5 = mucho"
        )
        goout = st.slider(
            "Frecuencia de salir con amigos",
            1, 5, 2,
            help="1 = casi nunca, 5 = muy frecuente"
        )

    # -------- Columna 6 --------
    with col6:
        Dalc = st.slider(
            "Consumo de alcohol (días de semana)",
            1, 5, 1,
            help="1 = muy bajo, 5 = muy alto"
        )
        Walc = st.slider(
            "Consumo de alcohol (fin de semana)",
            1, 5, 1,
            help="1 = muy bajo, 5 = muy alto"
        )
        health = st.slider(
            "Estado de salud actual",
            1, 5, 4,
            help="1 = muy malo, 5 = muy bueno"
        )
        absences = st.number_input(
            "Número de inasistencias",
            min_value=0,
            max_value=100,
            value=0
        )

    submitted = st.form_submit_button("Predecir atraso")

# ==============================
#  Predicción
# ==============================
if submitted:
    # Mapear selecciones en español a códigos originales del dataset
    school = SCHOOL_OPTS[school_es]
    sex = SEX_OPTS[sex_es]
    address = ADDRESS_OPTS[address_es]
    famsize = FAMSIZE_OPTS[famsize_es]
    Pstatus = PSTATUS_OPTS[Pstatus_es]
    Mjob = MJOB_OPTS[Mjob_es]
    Fjob = FJOB_OPTS[Fjob_es]
    reason = REASON_OPTS[reason_es]
    guardian = GUARDIAN_OPTS[guardian_es]
    schoolsup = SCHOOLSUP_OPTS[schoolsup_es]
    famsup = FAMSUP_OPTS[famsup_es]
    paid = PAID_OPTS[paid_es]
    activities = ACTIVITIES_OPTS[activities_es]
    nursery = NURSERY_OPTS[nursery_es]
    higher = HIGHER_OPTS[higher_es]
    internet = INTERNET_OPTS[internet_es]
    romantic = ROMANTIC_OPTS[romantic_es]

    # Construir el diccionario con los nombres EXACTOS de las columnas del dataset
    data = {
        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
        "Medu": Medu,
        "Fedu": Fedu,
        "Mjob": Mjob,
        "Fjob": Fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": Dalc,
        "Walc": Walc,
        "health": health,
        "absences": absences,
    }

    df = pd.DataFrame([data])

    # Probabilidad de ATRASO (clase 1)
    proba_atraso = winner_pipe.predict_proba(df)[0, 1]
    pred_int = int(proba_atraso >= BEST_THR)
    pred_label = REV_LABEL[pred_int]

    st.subheader("Resultado")
    st.write(f"**Predicción del modelo:** {pred_label}")
    st.write(f"**Probabilidad estimada de ATRASO:** {proba_atraso:.3f}")
    st.progress(float(proba_atraso))

    if pred_int == 1:
        st.warning("Este estudiante está en **riesgo de atraso escolar** según el modelo.")
    else:
        st.success("Este estudiante **no** está en riesgo de atraso escolar según el modelo.")
