📊 Proyecto de Machine Learning: Predicción de Abandono de Clientes (Churn)

🔗 Acceso al Dashboard (Streamlit App)

¡Haz clic aquí para ver la aplicación desplegada!
(Reemplaza el enlace de ejemplo con la URL que obtengas de Streamlit Cloud)

📸 Vista Previa del Dashboard
<img width="1142" height="508" alt="Captura de pantalla_2025-10-02_16-44-49" src="https://github.com/user-attachments/assets/92bab0de-6aa8-410f-b766-45aae69dbec6" />


🎯 Descripción y Objetivo del Proyecto

Este proyecto se centra en construir un modelo de Machine Learning capaz de predecir el abandono de clientes (Churn) en una compañía de telecomunicaciones.

El objetivo principal es identificar a los clientes con alto riesgo de irse con suficiente antelación, permitiendo a los equipos de negocio ejecutar estrategias de retención focalizadas y eficientes.

Enfoque Metodológico:

    Análisis Exploratorio (EDA): Identificación de los principales factores de riesgo (Mes a Mes, Baja Antigüedad, Fibra Óptica).

    Preprocesamiento: Codificación One-Hot para categóricas y estandarización para numéricas.

    Modelado: Uso de Regresión Logística por su alta interpretabilidad en el negocio.

💡 El Desafío Clave: El Desbalance de Clases

El dataset presentaba un severo desbalance de clases (muchos más clientes se quedan que los que abandonan), lo cual sesgó el modelo inicial, haciéndolo demasiado "pesimista" (alto Recall / baja Precision).

Para corregir esto, aplicamos una técnica de Resampling avanzada:

Solución: SMOTE (Synthetic Minority Over-sampling Technique)

SMOTE generó muestras sintéticas para la clase minoritaria ("Churn: Sí"). Esto forzó al modelo a aprender los patrones reales de abandono, corrigiendo el sesgo y llevando el modelo a un punto óptimo de equilibrio entre sensibilidad y precisión.
