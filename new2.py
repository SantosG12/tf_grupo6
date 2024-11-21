import streamlit as st
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler

# Título de la aplicación
st.title("Predicción de Problemas Cardiovasculares")
st.write("Ingrese los datos del paciente para predecir la probabilidad de problemas cardiovasculares.")

# Cargar el modelo y preprocesador
try:
    modelo_cargado = joblib.load('modelo_ganador_rf.pkl')  # Cargar el modelo
    preprocessor = joblib.load('preprocessor.pkl')  # Cargar el preprocesador
    st.success("Modelo y preprocesador cargados exitosamente.")
except Exception as e:
    st.error(f"Error al cargar el modelo o preprocesador: {e}")
    st.stop()

# Interfaz para ingresar datos manualmente
st.sidebar.header("Ingrese los datos del paciente")

genero = st.sidebar.selectbox("Género", ["Male", "Female"])
colesterol = st.sidebar.selectbox("Colesterol (1=Normal, 2=Elevado, 3=Muy Elevado)", ['1', '2', '3'])
glucosa = st.sidebar.selectbox("Glucosa (1=Normal, 2=Elevada, 3=Muy Elevada)", ['1', '2', '3'])
fuma = st.sidebar.radio("¿Fuma?", ['0', '1'])
toma_alcohol = st.sidebar.radio("¿Toma Alcohol?", ['0', '1'])
actividad_fisica = st.sidebar.radio("¿Realiza Actividad Física?", ['0', '1'])
edad = st.sidebar.number_input("Edad (años)", min_value=1, max_value=120, value=47, step=1)
altura = st.sidebar.number_input("Altura (cm)", min_value=50, max_value=250, value=156, step=1)
peso = st.sidebar.number_input("Peso (kg)", min_value=10, max_value=300, value=50, step=1)
presion_sistolica = st.sidebar.number_input("Presión Sistólica (mmHg)", min_value=50, max_value=300, value=100, step=1)
presion_diastolica = st.sidebar.number_input("Presión Diastólica (mmHg)", min_value=30, max_value=200, value=60, step=1)

# Calcular BMI
bmi = round(peso / ((altura / 100) ** 2), 2)
st.sidebar.markdown(f"### BMI Calculado: **{bmi}**")

# Crear un nuevo conjunto de datos con las columnas completas
nuevos_datos = pd.DataFrame({
    'Genero': [genero],
    'Colesterol': [colesterol],
    'Glucosa': [glucosa],
    'Fuma': [fuma],
    'Toma_alchol': [toma_alcohol],
    'Actividad_fisica': [actividad_fisica],
    'Edad': [edad],
    'Altura': [altura],
    'Peso': [peso],
    'Presion_arterial_sistolica': [presion_sistolica],
    'Presion_arterial_diastolica': [presion_diastolica],
    'Bmi': [bmi]
})

# Transformar los datos con el preprocesador
try:
    nuevos_datos_transformados = preprocessor.transform(nuevos_datos)
except Exception as e:
    st.error(f"Error al transformar los datos ingresados: {e}")
    st.stop()

# Realizar la predicción
try:
    prediccion = modelo_cargado.predict(nuevos_datos_transformados)
    probabilidad = modelo_cargado.predict_proba(nuevos_datos_transformados)[:, 1]
except Exception as e:
    st.error(f"Error al realizar la predicción: {e}")
    st.stop()

# Mostrar el resultado
st.subheader("Resultado de la Predicción")
if prediccion[0] == 1 and probabilidad[0] > 0.5:
    st.error("El modelo predice que el paciente tiene riesgo de problemas cardiovasculares.")
else:
    st.success("El modelo predice que el paciente NO tiene riesgo de problemas cardiovasculares.")

st.write(f"Probabilidad estimada de riesgo: {probabilidad[0]:.2%}")
