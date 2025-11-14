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
        school = st.selectbox(
            "Colegio (school)",
            ["GP", "MS"],
            help="GP = Gabriel Pereira, MS = Mousinho da Silveira"
        )
        sex = st.selectbox(
            "Sexo (sex)",
            ["F", "M"],
            help="F = Femenino, M = Masculino"
        )
        age = st.number_input(
            "Edad (age)",
            min_value=15,
            max_value=25,
            value=17
        )
        address = st.selectbox(
            "Tipo de domicilio (address)",
            ["U", "R"],
            help="U = Urbano, R = Rural"
        )
        famsize = st.selectbox(
            "Tamaño de familia (famsize)",
            ["LE3", "GT3"],
            help="LE3 = ≤ 3 miembros, GT3 = > 3 miembros"
        )
        Pstatus = st.selectbox(
            "Estado de convivencia de padres (Pstatus)",
            ["T", "A"],
            help="T = Juntos, A = Separados"
        )

    # -------- Columna 2 --------
    with col2:
        Medu = st.slider(
            "Educación de la madre (Medu)",
            0, 4, 2,
            help="0 = ninguna, 1 = primaria, 2 = 5º-9º, 3 = secundaria, 4 = superior"
        )
        Fedu = st.slider(
            "Educación del padre (Fedu)",
            0, 4, 2,
            help="0 = ninguna, 1 = primaria, 2 = 5º-9º, 3 = secundaria, 4 = superior"
        )
        Mjob = st.selectbox(
            "Trabajo de la madre (Mjob)",
            ["teacher", "health", "services", "at_home", "other"],
            help="teacher, health, services, at_home, other"
        )
        Fjob = st.selectbox(
            "Trabajo del padre (Fjob)",
            ["teacher", "health", "services", "at_home", "other"],
            help="teacher, health, services, at_home, other"
        )
        reason = st.selectbox(
            "Razón para elegir el colegio (reason)",
            ["home", "reputation", "course", "other"],
            help="home = cercano, reputation = reputación, course = curso, other = otro"
        )
        guardian = st.selectbox(
            "Apoderado principal (guardian)",
            ["mother", "father", "other"]
        )

    # -------- Columna 3 --------
    with col3:
        traveltime = st.slider(
            "Tiempo de traslado al colegio (traveltime)",
            1, 4, 1,
            help="1:<15m, 2:15-30m, 3:30-60m, 4:>1h"
        )
        studytime = st.slider(
            "Horas de estudio semanal (studytime)",
            1, 4, 2,
            help="1:<2h, 2:2-5h, 3:5-10h, 4:>10h"
        )
        failures = st.slider(
            "Nº de repeticiones previas (failures)",
            0, 4, 0,
            help="Número de veces que repitió curso/asignatura"
        )
        schoolsup = st.selectbox(
            "Apoyo educativo extra del colegio (schoolsup)",
            ["yes", "no"]
        )
        famsup = st.selectbox(
            "Apoyo educativo de la familia (famsup)",
            ["yes", "no"]
        )
        paid = st.selectbox(
            "Clases pagadas extra (paid)",
            ["yes", "no"]
        )

    col4, col5, col6 = st.columns(3)

    # -------- Columna 4 --------
    with col4:
        activities = st.selectbox(
            "Actividades extracurriculares (activities)",
            ["yes", "no"]
        )
        nursery = st.selectbox(
            "Asistió a inicial (nursery)",
            ["yes", "no"]
        )
        higher = st.selectbox(
            "Desea educación superior (higher)",
            ["yes", "no"]
        )
        internet = st.selectbox(
            "Acceso a Internet en casa (internet)",
            ["yes", "no"]
        )
        romantic = st.selectbox(
            "Tiene relación romántica (romantic)",
            ["yes", "no"]
        )

    # -------- Columna 5 --------
    with col5:
        famrel = st.slider(
            "Calidad de relaciones familiares (famrel)",
            1, 5, 4,
            help="1 = muy mala, 5 = excelente"
        )
        freetime = st.slider(
            "Tiempo libre después de clase (freetime)",
            1, 5, 3,
            help="1 = muy poco, 5 = mucho"
        )
        goout = st.slider(
            "Salir con amigos (goout)",
            1, 5, 2,
            help="1 = casi nunca, 5 = muy frecuente"
        )

    # -------- Columna 6 --------
    with col6:
        Dalc = st.slider(
            "Consumo de alcohol en días de semana (Dalc)",
            1, 5, 1,
            help="1 = muy bajo, 5 = muy alto"
        )
        Walc = st.slider(
            "Consumo de alcohol en fin de semana (Walc)",
            1, 5, 1,
            help="1 = muy bajo, 5 = muy alto"
        )
        health = st.slider(
            "Estado de salud actual (health)",
            1, 5, 4,
            help="1 = muy malo, 5 = muy bueno"
        )
        absences = st.number_input(
            "Número de inasistencias (absences)",
            min_value=0,
            max_value=100,
            value=0
        )

    submitted = st.form_submit_button("Predecir atraso")

# ==============================
#  Predicción
# ==============================
if submitted:
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
