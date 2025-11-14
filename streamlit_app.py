import streamlit as st
import joblib
import os
import pandas as pd  # lo usas al crear el DataFrame

# Carpeta donde está el modelo
ART_DIR = "artefactos"

@st.cache_resource
def load_pipeline():
    # 🔴 PON AQUÍ EL NOMBRE EXACTO DE TU ARCHIVO DE MODELO
    # Ejemplos:
    #   "modelo_pima.pkl"
    #   "modelo_atraso.joblib"
    model_path = os.path.join(ART_DIR, "modelo_atraso.joblib")
    return joblib.load(model_path)

winner_pipe = load_pipeline()

# Mapa de etiquetas fijo para tu problema
LABEL_MAP = {"NO_ATRASO": 0, "ATRASO": 1}
REV_LABEL = {v: k for k, v in LABEL_MAP.items()}  # {0:"NO_ATRASO",1:"ATRASO"}

BEST_THR = 0.5  # mismo que en tu entrenamiento

# ----------------- Interfaz -----------------
st.title("Predicción de atraso escolar por hábitos")

st.write(
    "Modelo de clasificación para predecir si un estudiante está en "
    "**ATRASO (1)** o **NO_ATRASO (0)** usando sus hábitos y contexto."
)

with st.form("form_atraso"):
    st.subheader("Datos del estudiante")

    col1, col2, col3 = st.columns(3)

    with col1:
        school = st.selectbox("school", ["GP", "MS"])
        sex = st.selectbox("sex", ["F", "M"])
        age = st.number_input("age", 15, 25, 17)
        address = st.selectbox("address", ["U", "R"])
        famsize = st.selectbox("famsize", ["LE3", "GT3"])
        Pstatus = st.selectbox("Pstatus", ["T", "A"])

    with col2:
        Medu = st.slider("Medu (educ. madre)", 0, 4, 2)
        Fedu = st.slider("Fedu (educ. padre)", 0, 4, 2)
        Mjob = st.selectbox("Mjob", ["teacher", "health", "services", "at_home", "other"])
        Fjob = st.selectbox("Fjob", ["teacher", "health", "services", "at_home", "other"])
        reason = st.selectbox("reason", ["home", "reputation", "course", "other"])
        guardian = st.selectbox("guardian", ["mother", "father", "other"])

    with col3:
        traveltime = st.slider("traveltime", 1, 4, 1)
        studytime = st.slider("studytime", 1, 4, 2)
        failures = st.slider("failures", 0, 4, 0)
        schoolsup = st.selectbox("schoolsup", ["yes", "no"])
        famsup = st.selectbox("famsup", ["yes", "no"])
        paid = st.selectbox("paid", ["yes", "no"])

    col4, col5, col6 = st.columns(3)
    with col4:
        activities = st.selectbox("activities", ["yes", "no"])
        nursery = st.se
