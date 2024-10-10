import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from weather_data_processing import preprocess_data  # Імпорт функції попередньої обробки
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Dict, Any

# Список локацій
locations = [
    'Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree',
    'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond',
    'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown',
    'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat',
    'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura',
    'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
    'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa',
    'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport',
    'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston',
    'AliceSprings', 'Darwin', 'Katherine', 'Uluru'
]

# Список напрямків вітру
wind_gust_dirs = ['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE',
                 'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW']

wind_dir_9am = ['W', 'NNW', 'SE', 'ENE', 'SW', 'SSE', 'S', 'NE', 'SSW',
               'N', 'WSW', 'ESE', 'E', 'NW', 'WNW', 'NNE']

wind_dir_3pm = ['WNW', 'WSW', 'E', 'NW', 'W', 'SSE', 'ESE', 'ENE',
               'NNW', 'SSW', 'SW', 'SE', 'N', 'S', 'NNE', 'NE']

# Статистики для налаштування повзунків
stats = {
    'MinTemp': {'min': -8.5, 'max': 33.9, 'step': 0.5, 'default': 12.0},
    'MaxTemp': {'min': -4.8, 'max': 48.1, 'step': 0.5, 'default': 23.2},
    'Rainfall': {'min': 0.0, 'max': 371.0, 'step': 0.5, 'default': 2.36},
    'Evaporation': {'min': 0.0, 'max': 145.0, 'step': 0.5, 'default': 5.47},
    'Sunshine': {'min': 0.0, 'max': 14.5, 'step': 0.5, 'default': 7.61},
    'WindGustSpeed': {'min': 6.0, 'max': 135.0, 'step': 1.0, 'default': 40.0},
    'WindSpeed9am': {'min': 0.0, 'max': 130.0, 'step': 1.0, 'default': 14.0},
    'WindSpeed3pm': {'min': 0.0, 'max': 87.0, 'step': 1.0, 'default': 18.66},
    'Humidity9am': {'min': 0.0, 'max': 100.0, 'step': 1.0, 'default': 68.88},
    'Humidity3pm': {'min': 0.0, 'max': 100.0, 'step': 1.0, 'default': 51.54},
    'Pressure9am': {'min': 950.0, 'max': 1041.0, 'step': 0.1, 'default': 1017.65},
    'Pressure3pm': {'min': 950.0, 'max': 1039.6, 'step': 0.1, 'default': 1015.26},
    'Cloud9am': {'min': 0.0, 'max': 9.0, 'step': 1.0, 'default': 4.44},
    'Cloud3pm': {'min': 0.0, 'max': 9.0, 'step': 1.0, 'default': 4.51},
    'Temp9am': {'min': -7.2, 'max': 40.2, 'step': 0.5, 'default': 16.99},
    'Temp3pm': {'min': -5.4, 'max': 46.7, 'step': 0.5, 'default': 21.68},
}

def predict(location, min_temp, max_temp, rainfall, evaporation, sunshine, 
             wind_gust_speed, wind_speed_9am, wind_speed_3pm,
             humidity_9am, humidity_3pm, pressure_9am, pressure_3pm,
             cloud_9am, cloud_3pm, temp_9am, temp_3pm, rain_today, 
             wind_gust_dir, wind_dir_9am, wind_dir_3pm):
    # Завантажуємо натреновану модель
    model = joblib.load('models/aussie_rain.joblib')
    
    # Формуємо дані для прогнозу
    data = np.expand_dims(np.array([
        location, min_temp, max_temp, rainfall, evaporation, sunshine, 
        wind_gust_speed, wind_speed_9am, wind_speed_3pm,
        humidity_9am, humidity_3pm, pressure_9am, pressure_3pm,
        cloud_9am, cloud_3pm, temp_9am, temp_3pm, rain_today,
        wind_gust_dir, wind_dir_9am, wind_dir_3pm
    ]), axis=0)
    
    # Здійснюємо прогнозування
    predictions = model.predict(data)
    
    return predictions[0]

# Заголовок застосунку
st.title('Прогнозування дощів')
st.markdown('Ця модель прогнозує, чи буде дощ завтра на основі даних про погоду.')
st.image('images/weather.png')

# Заголовок секції з характеристиками погоди
st.header("Характеристики погоди")

# Розподіл на колонки для кращого вигляду
col1, col2 = st.columns(2)

# Введення характеристик погоди
with col1:
    st.subheader("Температура, сонячне світло та вітер")
    min_temp = st.slider('Мінімальна температура (°C)', 
                         min_value=stats['MinTemp']['min'], 
                         max_value=stats['MinTemp']['max'], 
                         step=stats['MinTemp']['step'], 
                         value=stats['MinTemp']['default'])
    max_temp = st.slider('Максимальна температура (°C)', 
                         min_value=stats['MaxTemp']['min'], 
                         max_value=stats['MaxTemp']['max'], 
                         step=stats['MaxTemp']['step'], 
                         value=stats['MaxTemp']['default'])
    rainfall = st.slider('Кількість опадів (мм)', 
                         min_value=stats['Rainfall']['min'], 
                         max_value=stats['Rainfall']['max'], 
                         step=stats['Rainfall']['step'], 
                         value=stats['Rainfall']['default'])

    evaporation = st.slider('Випаровування (мм)', 
                            min_value=stats['Evaporation']['min'], 
                            max_value=stats['Evaporation']['max'], 
                            step=stats['Evaporation']['step'], 
                            value=stats['Evaporation']['default'])

    sunshine = st.slider('Сонячне світло (години)', 
                         min_value=stats['Sunshine']['min'], 
                         max_value=stats['Sunshine']['max'], 
                         step=stats['Sunshine']['step'], 
                         value=stats['Sunshine']['default'])
    wind_gust_speed = st.slider('Швидкість пориву вітру (км/год)', 
                                 min_value=stats['WindGustSpeed']['min'], 
                                 max_value=stats['WindGustSpeed']['max'], 
                                 step=stats['WindGustSpeed']['step'], 
                                 value=stats['WindGustSpeed']['default'])
    
    wind_gust_dir = st.selectbox('Напрямок пориву вітру', wind_gust_dirs, key='wind_gust_dir')
    wind_dir_9am = st.selectbox('Напрямок вітру о 9am', wind_dir_9am, key='wind_dir_9am')
    wind_dir_3pm = st.selectbox('Напрямок вітру о 3pm', wind_dir_3pm, key='wind_dir_3pm')
    location = st.selectbox('Виберіть локацію', locations, key='location')

with col2:
    st.subheader("Швидкість вітру та вологість")
    wind_speed_9am = st.slider('Швидкість вітру о 9am (км/год)', 
                                min_value=stats['WindSpeed9am']['min'], 
                                max_value=stats['WindSpeed9am']['max'], 
                                step=stats['WindSpeed9am']['step'], 
                                value=stats['WindSpeed9am']['default'])
    wind_speed_3pm = st.slider('Швидкість вітру о 3pm (км/год)', 
                                min_value=stats['WindSpeed3pm']['min'], 
                                max_value=stats['WindSpeed3pm']['max'], 
                                step=stats['WindSpeed3pm']['step'], 
                                value=stats['WindSpeed3pm']['default'])
    humidity_9am = st.slider('Вологість 9am (%)', 
                              min_value=stats['Humidity9am']['min'], 
                              max_value=stats['Humidity9am']['max'], 
                              step=stats['Humidity9am']['step'], 
                              value=stats['Humidity9am']['default'])
    humidity_3pm = st.slider('Вологість 3pm (%)', 
                              min_value=stats['Humidity3pm']['min'], 
                              max_value=stats['Humidity3pm']['max'], 
                              step=stats['Humidity3pm']['step'], 
                              value=stats['Humidity3pm']['default'])

    pressure_9am = st.slider('Тиск 9am (гПа)', 
                              min_value=stats['Pressure9am']['min'], 
                              max_value=stats['Pressure9am']['max'], 
                              step=stats['Pressure9am']['step'], 
                              value=stats['Pressure9am']['default'])
    pressure_3pm = st.slider('Тиск 3pm (гПа)', 
                             min_value=stats['Pressure3pm']['min'], 
                             max_value=stats['Pressure3pm']['max'], 
                             step=stats['Pressure3pm']['step'], 
                             value=stats['Pressure3pm']['default'])

    cloud_9am = st.slider('Хмарність о 9am', 
                          min_value=stats['Cloud9am']['min'], 
                          max_value=stats['Cloud9am']['max'], 
                          step=stats['Cloud9am']['step'], 
                          value=stats['Cloud9am']['default'])
    cloud_3pm = st.slider('Хмарність о 3pm', 
                          min_value=stats['Cloud3pm']['min'], 
                          max_value=stats['Cloud3pm']['max'], 
                          step=stats['Cloud3pm']['step'], 
                          value=stats['Cloud3pm']['default'])
    temp_9am = st.slider('Температура о 9am (°C)', 
                         min_value=stats['Temp9am']['min'], 
                         max_value=stats['Temp9am']['max'], 
                         step=stats['Temp9am']['step'], 
                         value=stats['Temp9am']['default'])
    temp_3pm = st.slider('Температура о 3pm (°C)', 
                         min_value=stats['Temp3pm']['min'], 
                         max_value=stats['Temp3pm']['max'], 
                         step=stats['Temp3pm']['step'], 
                         value=stats['Temp3pm']['default'])

# Дата
current_date = datetime.now().strftime('%Y-%m-%d')
st.write(f"Поточна дата: {current_date}")

# Введення RainToday
rain_today = st.radio('Чи йшов дощ сьогодні?', ('No', 'Yes'))

# Збір всіх характеристик у DataFrame
data = pd.DataFrame({
    'Location': [location],
    'MinTemp': [min_temp],
    'MaxTemp': [max_temp],
    'Rainfall': [rainfall],
    'Evaporation': [evaporation],
    'Sunshine': [sunshine],
    'WindGustSpeed': [wind_gust_speed],
    'WindSpeed9am': [wind_speed_9am],
    'WindSpeed3pm': [wind_speed_3pm],
    'Humidity9am': [humidity_9am],
    'Humidity3pm': [humidity_3pm],
    'Pressure9am': [pressure_9am],
    'Pressure3pm': [pressure_3pm],
    'Cloud9am': [cloud_9am],  
    'Cloud3pm': [cloud_3pm],  
    'Temp9am': [temp_9am],  
    'Temp3pm': [temp_3pm], 
    'WindGustDir': [wind_gust_dir],
    'WindDir9am': [wind_dir_9am],
    'WindDir3pm': [wind_dir_3pm],
    'RainToday': [rain_today],
})

# Виведення DataFrame для перевірки
st.write(data)

# Завантаження даних
data2 = joblib.load('aussie_rain.joblib')
input_cols = data2['input_cols']
categorical_cols = data2['categorical_cols']  # категоріальні ознаки
numeric_cols = data2['numeric_cols']          # числові ознаки
encoded_cols = data2['encoded_cols']          # закодовані ознаки
encoder = data2['encoder']                    # OneHotEncoder
imputer = data2['imputer']                    # SimpleImputer
scaler = data2['scaler']                      # MinMaxScaler
model = data2['model']                        # Модель RandomForest

data = data[input_cols]

def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    X_input = input_df[numeric_cols + encoded_cols]
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    return pred, prob

# Кнопка для прогнозування
if st.button("Прогнозувати ймовірність опадів"):
    # Викликаємо функцію прогнозування
    result, probability = predict_input({
        'Location': location,
        'MinTemp': min_temp,
        'MaxTemp': max_temp,
        'Rainfall': rainfall,
        'Evaporation': evaporation,
        'Sunshine': sunshine,
        'WindGustSpeed': wind_gust_speed,
        'WindSpeed9am': wind_speed_9am,
        'WindSpeed3pm': wind_speed_3pm,
        'Humidity9am': humidity_9am,
        'Humidity3pm': humidity_3pm,
        'Pressure9am': pressure_9am,
        'Pressure3pm': pressure_3pm,
        'Cloud9am': cloud_9am,
        'Cloud3pm': cloud_3pm,
        'Temp9am': temp_9am,
        'Temp3pm': temp_3pm,
        'RainToday': rain_today,
        'WindGustDir': wind_gust_dir,
        'WindDir9am': wind_dir_9am,
        'WindDir3pm': wind_dir_3pm
    })
    st.write(f"Прогнозована ймовірність опадів: {probability:.2f} (клас: {result})")