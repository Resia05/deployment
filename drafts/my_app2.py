import streamlit as st
import pandas as pd
import numpy as np
import joblib

def predict(min_temp, max_temp, rainfall, evaporation, sunshine, wind_gust_speed, humidity_9am, humidity_3pm, pressure_9am, pressure_3pm):
    model = joblib.load('models/aussie_rain.joblib')  # Завантаження моделі
    data = np.expand_dims(np.array([min_temp, max_temp, rainfall, evaporation, sunshine, wind_gust_speed,
                                     humidity_9am, humidity_3pm, pressure_9am, pressure_3pm]), axis=0)
    predictions = model.predict(data)
    return predictions[0]

# Заголовок застосунку
st.title('Прогнозування дощів')
st.markdown('Ця модель прогнозує, чи буде дощ завтра на основі даних про погоду.')
st.image('images/weather.png')
    
# Заголовок секції з характеристиками погоди
st.header("Характеристики погоди")
col1, col2 = st.columns(2)

# Введення характеристик погоди
with col1:
    st.text("Характеристики температури")
    min_temp = st.slider('Мінімальна температура (°C)', -10.0, 40.0, 0.5)
    max_temp = st.slider('Максимальна температура (°C)', -10.0, 50.0, 0.5)
    rainfall = st.slider('Кількість опадів (мм)', 0.0, 50.0, 0.5)

# Введення інших характеристик погоди
with col2:
    st.text("Інші характеристики")
    evaporation = st.slider('Випаровування (мм)', 0.0, 20.0, 0.5)
    sunshine = st.slider('Сонячне світло (години)', 0.0, 12.0, 0.5)
    wind_gust_speed = st.slider('Швидкість пориву вітру (км/год)', 0.0, 100.0, 1.0)
    humidity_9am = st.slider('Вологість 9am (%)', 0.0, 100.0, 1.0)
    humidity_3pm = st.slider('Вологість 3pm (%)', 0.0, 100.0, 1.0)
    pressure_9am = st.slider('Тиск 9am (гПа)', 950.0, 1050.0, 1.0)
    pressure_3pm = st.slider('Тиск 3pm (гПа)', 950.0, 1050.0, 1.0)

# Кнопка для прогнозування
if st.button("Прогнозувати дощ"):
    # Формуємо DataFrame з вхідними даними
    input_data = pd.DataFrame({
        'MinTemp': [min_temp],
        'MaxTemp': [max_temp],
        'Rainfall': [rainfall],
        'Evaporation': [evaporation],
        'Sunshine': [sunshine],
        'WindGustSpeed': [wind_gust_speed],
        'Humidity9am': [humidity_9am],
        'Humidity3pm': [humidity_3pm],
        'Pressure9am': [pressure_9am],
        'Pressure3pm': [pressure_3pm]
    })

    # Викликаємо функцію попередньої обробки
    from weather_data_processing import preprocess_data
    processed_data = preprocess_data(input_data)

    # Викликаємо функцію прогнозування
    result = predict(processed_data)

    # Виводимо прогноз
    st.write(f"Прогнозований дощ завтра: {result}")

