import streamlit as st
import pandas as pd
import joblib
import json
import os

# ----------------- Cargar artefactos -----------------
ART_DIR = os.path.join("artefactos", "v1")

@st.cache_resource
def load_pipeline():
    # ajusta el nombre del archivo si es distinto
    # por ejemplo: pipeline_RFS.joblib, pipeline_LRN.joblib, etc.
    pipe_path = os.path.join(ART_DIR, "pipeline_RFS.joblib")
    model = joblib.load(pipe_path)
    return model

@st.cache_resource
def load_label_map():
    with open(os.path.join(ART_DIR, "label_map.json"), "r", encoding="utf-8") as f:
        return json.load(f)

winner_pipe = load_pipeline()
LABEL_MAP = load_label_map()
REV_LABEL = {v: k for k, v in LABEL_MAP.items()}  # {0:"NO_ATRASO",1:"ATRASO"}

BEST_THR = 0.5  # mismo que en tu decision_policy

# ----------------- Interfaz -----------------
st.title("Predicción de atraso escolar por hábitos")

st.write("Modelo de clasificación para predecir si un estudiante está en **ATRASO (1)** "
         "o **NO_ATRASO (0)** usando sus hábitos y contexto.")

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
        nursery = st.selectbox("nursery", ["yes", "no"])
        higher = st.selectbox("higher", ["yes", "no"])
        internet = st.selectbox("internet", ["yes", "no"])
        romantic = st.selectbox("romantic", ["yes", "no"])

    with col5:
        famrel = st.slider("famrel", 1, 5, 4)
        freetime = st.slider("freetime", 1, 5, 3)
        goout = st.slider("goout", 1, 5, 2)

    with col6:
        Dalc = st.slider("Dalc (alcohol diario)", 1, 5, 1)
        Walc = st.slider("Walc (alcohol fin de semana)", 1, 5, 1)
        health = st.slider("health", 1, 5, 4)
        absences = st.number_input("absences", 0, 100, 0)

    submitted = st.form_submit_button("Predecir atraso")

# ----------------- Predicción -----------------
if submitted:
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
    proba_atraso = winner_pipe.predict_proba(df)[0, 1]
    pred_int = int(proba_atraso >= BEST_THR)
    pred_label = REV_LABEL[pred_int]

    st.subheader("Resultado")
    st.write(f"**Predicción:** {pred_label}")
    st.write(f"**Probabilidad de ATRASO:** {proba_atraso:.3f}")
    st.progress(float(proba_atraso))

    if pred_int == 1:
        st.warning("Este estudiante está en **riesgo de atraso** según el modelo.")
    else:
        st.success("Este estudiante **no** está en riesgo de atraso según el modelo.")
