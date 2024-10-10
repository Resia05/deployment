import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from weather_data_processing import preprocess_data  # Імпорт функції попередньої обробки

def predict(location, min_temp, max_temp, rainfall, evaporation, sunshine, wind_gust_speed, 
            humidity_9am, humidity_3pm, pressure_9am, pressure_3pm, rain_today, rain_tomorrow):
    model = joblib.load('models/aussie_rain.joblib')  # Завантаження моделі

    # Формуємо вхідні дані для моделі
    input_data = np.array([location, min_temp, max_temp, rainfall, evaporation, sunshine, wind_gust_speed,
                            humidity_9am, humidity_3pm, pressure_9am, pressure_3pm, rain_today, rain_tomorrow]).reshape(1, -1)

    # Додаємо стовпець 'Date' до DataFrame
    date_today = datetime.now().strftime('%Y-%m-%d')  # або використайте іншу дату
    df_input = pd.DataFrame(input_data, columns=['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 
                                                  'Evaporation', 'Sunshine', 'WindGustSpeed',
                                                  'Humidity9am', 'Humidity3pm', 
                                                  'Pressure9am', 'Pressure3pm', 
                                                  'RainToday', 'RainTomorrow'])

    # Додаємо стовпець 'Date' без перетворення
    df_input['Date'] = date_today  # Додаємо 'Date' як рядок

    # Задаємо тип стовпця 'Date' як object (необов'язково, оскільки pandas автоматично розпізнає тип)
    df_input['Date'] = df_input['Date'].astype(object)

    # Обробляємо дані (якщо потрібно)
    processed_data = preprocess_data(df_input)

    # Виконуємо прогноз
    predictions = model.predict(processed_data)
    return predictions[0]

# Заголовок застосунку
st.title('Прогнозування дощів')
st.markdown('Ця модель прогнозує, чи буде дощ завтра на основі даних про погоду.')
st.image('images/weather.png')

# Заголовок секції з характеристиками погоди
st.header("Характеристики погоди")
col1, col2 = st.columns(2)

# Введення для Location
location = st.selectbox('Локація', ['Location1', 'Location2', 'Location3'])  # Змініть на ваші локації

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

# Введення RainToday
rain_today = st.radio('Чи йшов дощ сьогодні?', ('Yes', 'No'))

# Введення RainTomorrow
rain_tomorrow = st.radio('Чи буде дощ завтра?', ('Yes', 'No'))

# Кнопка для прогнозування
if st.button("Прогнозувати дощ"):
    # Викликаємо функцію прогнозування
    result = predict(location, min_temp, max_temp, rainfall, evaporation, sunshine, wind_gust_speed,
                     humidity_9am, humidity_3pm, pressure_9am, pressure_3pm,
                     rain_today, rain_tomorrow)
    st.write(f"Прогнозований дощ завтра: {result}")  # Виводимо прогноз
