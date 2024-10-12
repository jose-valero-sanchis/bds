import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import os

# Cargar el modelo previamente entrenado
MODEL_PATH = 'best_model.h5'  # Cambia esta ruta a la del archivo de tu modelo
model = load_model(MODEL_PATH)

# Configurar la interfaz de usuario de Streamlit
st.title("Predicción de Diagnósticos de Tumores")
st.write("Sube un archivo .csv con los datos de concentración de metabolitos para predecir el diagnóstico.")

# Cargar el archivo .csv
uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")

if uploaded_file:
    # Leer el archivo CSV
    input_data = pd.read_csv(uploaded_file, sep = ";", usecols=range(1,16), decimal=',')

    def replace_comma(value):
        if isinstance(value, str):
            # Reemplaza la primera coma por un punto
            value = value.replace(',', '.', 1)
            # Elimina el resto de las comas
            value = value.replace(',', '')
        return value

    input_data = input_data.applymap(replace_comma).astype(float)
    
    # Verificar las columnas del archivo
    st.write("Datos cargados:")
    st.write(input_data.head())

    # Realizar predicciones
    predictions = model.predict(input_data)
    prediction_classes = np.argmax(predictions, axis=1)
    diagnosis_labels = ['Astrocytoma', 'Glioblastoma', 'Meningioma']  # Ajusta estos nombres según tu contexto

    # Crear un DataFrame de resultados
    results_df = input_data.copy()
    results_df['Prob_Astrocytoma'] = predictions[:, 0]
    results_df['Prob_Glioblastoma'] = predictions[:, 1]
    results_df['Prob_Meningioma'] = predictions[:, 2]
    results_df['Diagnóstico_Predicho'] = [diagnosis_labels[i] for i in prediction_classes]

    # Mostrar los resultados
    st.write("Resultados de las predicciones:")
    st.write(results_df)

    # Guardar los resultados en un archivo CSV
    output_filename = 'predicciones_tumor.csv'
    results_df.to_csv(output_filename, index=False)

    # Proveer el archivo para descargar
    with open(output_filename, 'rb') as f:
        st.download_button(label="Descargar resultados", data=f, file_name=output_filename, mime='text/csv')

    st.success("Predicciones completadas y archivo listo para descargar.")