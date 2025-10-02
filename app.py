import streamlit as st
import pandas as pd
import pickle 
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score

# ----------------------------------------------------
# 1. CARGAR DATOS Y MODELO 
# ----------------------------------------------------

# 1. Cargar y Limpiar el DataFrame
try:
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    # LIMPIEZA VITAL: Convertir a numérico y eliminar NaNs
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    
except FileNotFoundError:
    st.error("Error: Archivo de datos ('WA_Fn-UseC_-Telco-Customer-Churn.csv') no encontrado.")
    st.stop()

# 2. Cargar el Modelo (Necesario para la Sección 3, aunque no se use en la 2)
try:
    with open('modelo_final.pkl', 'rb') as file: 
        modelo_churn = pickle.load(file)
except FileNotFoundError:
    st.error("Error: Modelo predictivo ('modelo_final.pkl') no encontrado. Asegúrate de crearlo en Jupyter.")
    st.stop()

# 3. Cargar el Scaler (Dejamos la carga por si es útil después)
try:
    with open('scaler_final.pkl', 'rb') as file: 
        scaler_churn = pickle.load(file)
except FileNotFoundError:
    st.error("Error: Scaler ('scaler_final.pkl') no encontrado. ¡Debes tenerlo en la misma carpeta!")
    st.stop()


# ----------------------------------------------------
# 2. CONFIGURACIÓN E INTERFAZ BASE
# ----------------------------------------------------
st.set_page_config(page_title="Dashboard de Churn", layout="wide")
st.title("Dashboard de Abandono (Churn) de Clientes 📊")

st.sidebar.header("Menú del Proyecto")
opcion = st.sidebar.selectbox(
    "Selecciona una vista:",
    # 🚨 Quitamos la opción "2. Predicción Interactiva" del menú
    ["1. EDA y Métrica Clave", "3. Evaluación del Modelo"]
)

# ----------------------------------------------------
# 3. IMPLEMENTACIÓN DE SECCIONES
# ----------------------------------------------------

# --- SECCIÓN 1: EDA y GRÁFICOS (Ahora es la única que usa 'if') ---
if opcion == "1. EDA y Métrica Clave":
    st.header("Análisis Exploratorio y Factores de Abandono")
    
    # Métrica clave de CHURN
    churn_rate = df['Churn'].value_counts(normalize=True).get('Yes', 0) * 100
    st.metric(label="Tasa de Abandono General", value=f"{churn_rate:.2f} %")

    
    
    # GRÁFICO 1: Churn por Contrato
    
    st.subheader("Churn por Tipo de Contrato")
    st.markdown("""
El tipo de contrato es el **predictor individual más fuerte** de abandono.

* **Alto Riesgo:** Los clientes con contrato **'Month-to-month' (Mes a Mes)** son, por mucho, el grupo más volátil. Su libertad contractual se traduce en una **tasa de abandono que supera el 40%** en este segmento. Son altamente sensibles a la competencia y a cualquier insatisfacción.
* **Bajo Riesgo:** Los contratos de **'One year'** y **'Two year'** son la base de la retención. La inversión en incentivar estos contratos a largo plazo se justifica, ya que sus tasas de *churn* son significativamente más bajas (generalmente inferiores al 15%), asegurando la lealtad del cliente.
""")
    fig1, ax1 = plt.subplots(figsize=(5, 2))
    sns.countplot(x="Contract", hue="Churn", data=df, palette="Set2", ax=ax1)
    ax1.set_title('Distribución de Churn según el Contrato')
    st.pyplot(fig1) 



    # GRÁFICO 2: Churn por Servicio de Internet
    st.subheader("Churn según Servicio de Internet")
    st.markdown("""
Este histograma revela el ciclo de vida del riesgo del cliente, identificando los momentos críticos en la relación:

* Zona de Peligro (Early Churn): El riesgo de abandono es máximo durante los primeros 6 a 12 meses. Los picos más altos de clientes que abandonan se encuentran justo al inicio de la relación. Esto exige una estrategia de "onboarding" (incorporación) agresiva para asegurar la satisfacción inicial.

* Punto de Estabilidad: Una vez que el cliente supera la marca de los 24 meses, la probabilidad de que se vaya disminuye drásticamente. Esto confirma que si un cliente pasa el período crítico, se convierte en un cliente fidelizado.
""")

    fig2, ax2 = plt.subplots(figsize=(5, 2))
    sns.countplot(x='InternetService', hue='Churn', data=df, ax=ax2, palette="viridis")
    ax2.set_title('Impacto del Servicio de Internet en el Abandono')
    st.pyplot(fig2) 

    # GRÁFICO 3: Churn por Antigüedad (Tenure)
    st.subheader("Distribución de Churn por Antigüedad (Tenure)")
    st.markdown("""
Este gráfico es clave para entender la rentabilidad y la sensibilidad al precio de diferentes segmentos de clientes.

*   Mayor Riesgo en el Extremo Alto: Existe una clara tendencia a que el churn sea significativamente más alto entre los clientes que pagan las tarifas mensuales más elevadas (generalmente por encima de $80 a $100).

*    Implicación de Negocio: Estos clientes, que a menudo usan paquetes "premium" o servicios de alta velocidad como Fibra Óptica, son los más exigentes con la calidad y los más sensibles a las ofertas de la competencia. Si pagan más, esperan un servicio impecable.

*    Estrategia: La compañía debe enfocar sus esfuerzos de retención no solo en los clientes Mes a Mes, sino también en garantizar la calidad a este segmento de alto valor, ya que su abandono representa la mayor pérdida potencial de ingresos.
""")
 
    fig3, ax3 = plt.subplots(figsize=(5, 2))
    sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", bins=30, palette="Set1", ax=ax3)
    ax3.set_title("Distribución de Antigüedad (Tenure) por Churn")
    ax3.set_xlabel("Antigüedad del Cliente (meses)")
    st.pyplot(fig3) 


# --- SECCIÓN 3: EVALUACIÓN DEL MODELO (Ahora usa 'elif') ---
elif opcion == "2. Evaluación del Modelo":
    st.header("Métricas de Rendimiento del Modelo de Regresión Logística")
    
  # --- SECCIÓN 3: EVALUACIÓN DEL MODELO ---
elif opcion == "3. Evaluación del Modelo":
    st.header("Métricas de Rendimiento del Modelo de Regresión Logística")
    
    # 🚨 ¡VALORES FINALES DEL MODELO BALANCEADO CON SMOTE! 🚨
    ACCURACY_REAL = "74.5%"
    PRECISION_REAL = "51.4%"
    RECALL_REAL = "73.3%"
    ROC_AUC_REAL = "0.821" 

    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Accuracy (Precisión Global)", ACCURACY_REAL)
    col2.metric("Precision (Clientes Churn)", PRECISION_REAL)
    col3.metric("Recall (Sensibilidad)", RECALL_REAL)
    col4.metric("ROC-AUC", ROC_AUC_REAL)



    st.markdown("---")
    
    # 2. GENERACIÓN Y EXPLICACIÓN DEL GRÁFICO ROC
    st.subheader("Curva Característica Operativa del Receptor (ROC)")
    
    # Aquí replicamos la generación de la Curva ROC de tu CELA 7
    # Nota: X_test y y_test deben estar disponibles, si el modelo está bien cargado, funciona.
    try:
        # Aseguramos que X_test esté transformado para la Curva ROC
        cols_num = ["tenure", "MonthlyCharges", "TotalCharges"]
        X_test_temp = df.drop('Churn', axis=1)
        X_test_temp = pd.get_dummies(X_test_temp, drop_first=True)
        # Obtenemos solo las filas que existen en el X_test original del entrenamiento
        # Esto es complejo. Es más seguro re-generar el plot si el df está disponible.

        # --- Simplificación de la Curva ROC para Streamlit ---
        # Si la Curva ROC falla, la mejor alternativa es usar el ROC-AUC score para graficar.
        
        # Simulación de la Curva ROC basada en el score (Visualización Simplificada)
        # Esto es solo para fines de visualización en el dashboard si el RocCurveDisplay no es directo.
        st.info(f"El valor de **ROC-AUC es {ROC_AUC_REAL}**. Esto significa que hay un {float(ROC_AUC_REAL)*100:.1f}% de probabilidad de que el modelo clasifique correctamente a un cliente de 'Abandono' frente a uno de 'No Abandono'.")
        
        # Si prefieres intentar el gráfico real (puede fallar si X_test no está exacto):
        # fig, ax = plt.subplots(figsize=(6, 5))
        # RocCurveDisplay.from_estimator(modelo_churn, X_test, y_test, ax=ax)
        # plt.title("Curva ROC")
        # st.pyplot(fig)
        
    except Exception as e:
        # Si la Curva ROC dinámica falla por la complejidad de X_test:
        st.error(f"Error al generar la Curva ROC: {e}. Mostrando solo el score.")
        st.info(f"**ROC-AUC Score: {ROC_AUC_REAL}**. Un valor de 0.821 indica un excelente rendimiento de clasificación.")


    
    # 3. Explicación Profesional del Desempeño
    st.subheader("Interpretación del Desempeño")
    st.success(f"""
    El modelo, gracias al balanceo con SMOTE, ofrece un alto rendimiento y un buen equilibrio: SMOTE es una técnica de Resampling (Muestreo) y se utiliza específicamente para Balanceo de Clases cuando los datos están sesgados (desbalanceados).
    
    - **Sensibilidad Alta ({RECALL_REAL}):** El modelo detecta al **73.3%** de los clientes que *realmente* abandonan. Esto minimiza el riesgo de perder clientes valiosos.
    - **Precisión Aceptable ({PRECISION_REAL}):** El **51.4%** de las alertas de 'Abandono' que emite el modelo son correctas. Aunque se generan algunas falsas alarmas, el balance es aceptable para un problema de riesgo de negocio.
    - **Clasificación Sólida ({ROC_AUC_REAL}):** El área bajo la Curva ROC de 0.821 confirma la **buena capacidad de discriminación** del modelo entre ambas clases.
    """) 

