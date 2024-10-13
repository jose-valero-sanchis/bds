import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler
import joblib  

# Título de la aplicación
st.title("Predicción de Diagnóstico de Tumores Cerebrales")

st.write("""
Esta aplicación permite cargar datos de concentraciones de metabolitos para predecir el tipo de tumor cerebral.
""")

# Cargar el modelo guardado
@st.cache_resource
def load_saved_model():
    model = load_model('mejor_modelo.h5')
    return model

# Cargar el escalador guardado
@st.cache_resource
def load_scaler():
    scaler = joblib.load('robust_scaler.joblib')
    return scaler

model = load_saved_model()
scaler = load_scaler()

# Cargar el archivo CSV de entrada
uploaded_file = st.file_uploader("Carga un archivo CSV con las concentraciones de metabolitos", type=["csv"])

if uploaded_file is not None:
    # Leer el archivo CSV
    input_data = pd.read_csv(uploaded_file)
    
    st.write("Datos cargados:")
    st.write(input_data.head())

    # Verificar si las columnas necesarias están presentes
    required_columns = [f"Metabolite {i}" for i in range(15)]  # Lista de nombres de columnas de metabolitos esperados
    if all(col in input_data.columns for col in required_columns):
        # Preprocesar los datos
        X_input = input_data[required_columns]
        X_scaled = scaler.transform(X_input)
        
        # Realizar las predicciones
        predictions = model.predict(X_scaled)
        
        # Obtener las probabilidades y el diagnóstico recomendado
        diagnosis_classes = ['ASTROCYTOMA', 'GLIOBLASTOMA', 'MENINGIOMA']
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_labels = [diagnosis_classes[i] for i in predicted_classes]
        
        # Crear un DataFrame con los resultados
        results = input_data.copy()
        for idx, diagnosis in enumerate(diagnosis_classes):
            results[f'Probabilidad_{diagnosis}'] = predictions[:, idx]
        results['Diagnóstico_Predicho'] = predicted_labels
        
        st.write("Resultados de la predicción:")
        st.write(results.head())
        
        # Botón para descargar los resultados
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')
        
        csv = convert_df(results)
        
        st.download_button(
            label="Descargar resultados como CSV",
            data=csv,
            file_name='resultados_prediccion.csv',
            mime='text/csv',
        )
    else:
        st.error("El archivo CSV no contiene las columnas necesarias. Por favor, verifica el formato del archivo.")
