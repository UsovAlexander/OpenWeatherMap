import time
import warnings
import requests

import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import aiohttp
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from functools import wraps

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Анализ климатических данных",
    layout="wide"
)

st.title("Анализ температурных данных и мониторинг текущей температуры")
st.markdown("Анализ исторических температурных данных и мониторинг текущей погоды с помощью OpenWeatherMap API")

if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None

with st.sidebar:
    st.header("Настройки")

    api_key = st.text_input(
        "API Key OpenWeatherMap", 
        type="password",
        help="Введите ваш API ключ OpenWeatherMap"
    )
    
    uploaded_file = st.file_uploader("Загрузите исторические данные (CSV)", type=['csv'])
    st.session_state.uploaded_df = uploaded_file
    
    cities = ["Berlin", "Cairo", "Dubai", "Beijing", "Moscow"]
    selected_city = st.selectbox("Выберите город", cities)
    
    st.subheader("Параметры анализа")
    window_size = st.slider("Размер окна скользящего среднего (дни)", 7, 90, 30)
    sigma_threshold = st.slider("Порог аномалии (σ)", 1.0, 3.0, 2.0)
    
    use_parallel = st.checkbox("Использовать параллельные вычисления", value=True)
    
    request_method = st.radio(
        "Метод API-запросов",
        ["Синхронный (один город)", "Асинхронный (несколько городов)"]
    )

@st.cache_data
def load_and_process_data(file, city, window=30, sigma_threshold=2, use_parallel=False, test=False, df_size=1):
    if file is None:
        st.warning("Загрузите исторические данные!")
        return None, None
    
    if use_parallel:
        df = pl.read_csv(file)
        df = df.with_columns(pl.col('timestamp').str.to_datetime())
        city_data_pl = df.filter(pl.col('city') == city).sort('timestamp')
        
        if test:
            city_data_pl = pl.concat([city_data_pl] * df_size)
        
        city_data_pl = city_data_pl.with_columns([
            pl.col('temperature').rolling_mean(window_size=window, min_periods=1).alias('rolling_mean'),
            pl.col('temperature').rolling_std(window_size=window, min_periods=1).alias('rolling_std')
        ])
        
        city_data_pl = city_data_pl.with_columns([
            ((pl.col('temperature') - pl.col('rolling_mean')).abs() > 
             (sigma_threshold * pl.col('rolling_std')))
            .fill_null(False)
            .alias('is_anomaly')
        ])
        
        seasonal_stats_pl = city_data_pl.group_by('season').agg([
            pl.col('temperature').mean().alias('season_mean'),
            pl.col('temperature').std().alias('season_std'),
            pl.col('temperature').min().alias('season_min'),
            pl.col('temperature').max().alias('season_max')
        ]).sort('season')
        
        city_data = city_data_pl.to_pandas()
        seasonal_stats = seasonal_stats_pl.to_pandas()
        seasonal_stats.set_index('season', inplace=True)
        
    else:
        df = pd.read_csv(file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        city_data = df[df['city'] == city].copy().sort_values('timestamp')
        
        if test:
            city_data = pd.concat([city_data] * df_size)
        
        city_data['rolling_mean'] = city_data['temperature'].rolling(window=window, min_periods=1).mean()
        city_data['rolling_std'] = city_data['temperature'].rolling(window=window, min_periods=1).std()
        
        city_data['rolling_std_filled'] = city_data['rolling_std'].fillna(0)
        city_data['is_anomaly'] = (
            np.abs(city_data['temperature'] - city_data['rolling_mean']) > 
            (sigma_threshold * city_data['rolling_std_filled'])
        )
        
        city_data = city_data.drop(columns=['rolling_std_filled'])
        
        seasonal_stats = city_data.groupby('season')['temperature'].agg([
            ('season_mean', 'mean'),
            ('season_std', 'std'),
            ('season_min', 'min'),
            ('season_max', 'max')
        ]).round(2)
    
    return city_data, seasonal_stats

def get_current_weather_sync(city, api_key):
    if not api_key:
        return None, "API-ключ не предоставлен"
    
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            'q': city,
            'appid': api_key,
            'units': 'metric',
            'lang': 'ru'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 401:
            return None, '{"cod":401, "message": "Invalid API key. Please see https://openweathermap.org/faq#error401 for more info."}'
        elif response.status_code == 429:
            return None, "Превышен лимит запросов, попробуйте позже"
        elif response.status_code == 200:
            data = response.json()
            return {
                'temp': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'description': data['weather'][0]['description'],
                'icon': data['weather'][0]['icon']
            }, None
        else:
            return None, f"Ошибка API: {response.status_code}"
    except Exception as e:
        return None, f"Ошибка запроса: {str(e)}"

async def get_current_weather_async(city, api_key, session):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            'q': city,
            'appid': api_key,
            'units': 'metric',
            'lang': 'ru'
        }
        
        async with session.get(url, params=params, timeout=10) as response:
            if response.status == 401:
                return None, '{"cod":401, "message": "Invalid API key. Please see https://openweathermap.org/faq#error401 for more info."}'
            elif response.status == 429:
                return None, "Превышен лимит запросов, попробуйте позже"
            elif response.status == 200:
                data = await response.json()
                return {
                    'city': city,
                    'temp': data['main']['temp'],
                    'description': data['weather'][0]['description']
                }
            else:
                return {'city': city, 'error': f"Статус: {response.status}"}
    except Exception as e:
        return {'city': city, 'error': str(e)}

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper

def create_time_series_plot(df, city):
    fig = make_subplots(
        rows=2, 
        cols=1,
        subplot_titles=(
            f'{city} - Температурный временной ряд (аномалии отмечены красным)',
            'Скользящее среднее и доверительный интервал'
        ),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4],
        shared_xaxes=True
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['temperature'],
            mode='lines',
            name='Среднесуточная температура',
            line=dict(color='lightblue', width=1)
        ),
        row=1, 
        col=1
    )
    
    anomalies = df[df['is_anomaly']]
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies['timestamp'],
                y=anomalies['temperature'],
                mode='markers',
                name='Температурные аномалии',
                marker=dict(color='red', size=8, symbol='diamond'),
                hovertemplate='<b>Аномальная температура</b>: %{y:.1f}°C<br>Дата: %{x}<extra></extra>'
            ),
            row=1, 
            col=1
        )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['rolling_mean'],
            mode='lines',
            name=f'Скользящее среднее ({window_size} дней)',
            line=dict(color='blue', width=2)
        ),
        row=2, col=1
    )
    
    upper_bound = df['rolling_mean'] + sigma_threshold * df['rolling_std']
    lower_bound = df['rolling_mean'] - sigma_threshold * df['rolling_std']
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=upper_bound,
            mode='lines',
            name='Верхняя граница',
            line=dict(color='gray', width=1, dash='dash'),
            showlegend=False
        ),
        row=2, 
        col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=lower_bound,
            mode='lines',
            name='Нижняя граница',
            line=dict(color='gray', width=1, dash='dash'),
            showlegend=False
        ),
        row=2, 
        col=1
    )
    
    fig.update_layout(
        height=700,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Дата", row=2, col=1)
    fig.update_yaxes(title_text="Температура (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Температура (°C)", row=2, col=1)
    
    return fig

def create_seasonal_plot(seasonal_stats):
    seasons_order = ['winter', 'spring', 'summer', 'autumn']
    seasonal_stats = seasonal_stats.reindex(seasons_order)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=seasonal_stats.index,
        y=seasonal_stats['season_mean'],
        name='Средняя температура',
        marker_color='white'
    ))
    
    fig.add_trace(go.Scatter(
        x=seasonal_stats.index,
        y=seasonal_stats['season_max'],
        name='Максимальная температура',
        line=dict(color='red', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=seasonal_stats.index,
        y=seasonal_stats['season_min'],
        name='Минимальная температура',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Сезонное распределение температур',
        xaxis_title='Сезон',
        yaxis_title='Температура (°C)',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def main():
    if not st.session_state.uploaded_df:
        st.warning("Загрузите исторические данные!")
    else:
        with st.spinner('Анализируем данные...'):
            city_data, seasonal_stats = load_and_process_data(
                st.session_state.uploaded_df, 
                selected_city, 
                window_size,
                sigma_threshold,
                use_parallel
            )

            if city_data is not None and not city_data.empty:
                anomalies = city_data[city_data['is_anomaly']]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Средняя температура", f"{city_data['temperature'].mean():.1f}°C")
        with col2:
            st.metric("Максимальная температура", f"{city_data['temperature'].max():.1f}°C")
        with col3:
            st.metric("Минимальная температура", f"{city_data['temperature'].min():.1f}°C")
        with col4:
            anomaly_count = city_data['is_anomaly'].sum()
            anomaly_pct = (anomaly_count / len(city_data)) * 100
            st.metric("Аномальные дни", f"{anomaly_count}", f"{anomaly_pct:.1f}%")
        
        with st.expander("Просмотр данных"):
            st.dataframe(city_data[['timestamp', 'temperature', 'season', 'rolling_mean', 'is_anomaly']])
        
        tab1, tab2 = st.tabs(["Анализ временных рядов", "Сезонный анализ"])
        
        with tab1:
            fig1 = create_time_series_plot(city_data, selected_city)
            st.plotly_chart(fig1, width='stretch')
            
            if anomaly_count > 0:
                st.subheader("Анализ температурных аномалий")
                latest_anomaly = anomalies.iloc[-1] if not anomalies.empty else None
                
                if latest_anomaly is not None:
                    anomaly_temp = latest_anomaly['temperature']
                    anomaly_mean = latest_anomaly['rolling_mean']
                    deviation = anomaly_temp - anomaly_mean
                    deviation_sigma = deviation / latest_anomaly['rolling_std']
                    
                    st.info(
                        f"Последняя аномалия: **{latest_anomaly['timestamp'].strftime('%Y-%m-%d')}** - "
                        f"Температура: **{anomaly_temp:.1f}°C** | "
                        f"Отклонение от среднего: **{deviation:+.1f}°C** ({deviation_sigma:.1f}σ)"
                    )
        
        with tab2:
            fig2 = create_seasonal_plot(seasonal_stats)
            st.plotly_chart(fig2, width='stretch')

            st.subheader("Сезонная статистика")
            st.dataframe(seasonal_stats)
        
        st.divider()
        st.header("Мониторинг текущей погоды")
        
        if api_key:
            current_season = city_data.iloc[-1]['season'] if not city_data.empty else 'winter'
            season_stats = seasonal_stats.loc[current_season] if current_season in seasonal_stats.index else None
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if st.button("Получить текущую погоду", type="primary"):
                    if request_method == "Синхронный (один город)":
                        with st.spinner("Получаем данные о текущей погоде..."):
                            weather_data, error = get_current_weather_sync(selected_city, api_key)
                            
                            if error:
                                st.error(f"Ошибка: {error}")
                            elif weather_data:
                                current_temp = weather_data['temp']
                                
                                if season_stats is not None:
                                    is_normal = (
                                        abs(current_temp - season_stats['season_mean']) <= sigma_threshold * season_stats['season_std']
                                    )
                                else:
                                    is_normal = True
                                
                                temp_status = "Норма" if is_normal else "Аномалия"
                                st.metric("Текущая температура", f"{current_temp:.1f}°C", delta=temp_status)
                                
                                st.write(f"Ощущается как: {weather_data['feels_like']:.1f}°C")
                                st.write(f"Влажность: {weather_data['humidity']}%")
                                st.write(f"Описание: {weather_data['description']}")
                                
                                if season_stats is not None:
                                    st.info(
                                        f"Нормальный диапазон для {current_season}: "
                                        f"{season_stats['season_mean']:.1f}°C ± "
                                        f"{sigma_threshold * season_stats['season_std']:.1f}°C"
                                    )
                    
                    else:
                        async def fetch_all_cities():
                            async with aiohttp.ClientSession() as session:
                                tasks = []
                                for city in cities:
                                    task = asyncio.create_task(
                                        get_current_weather_async(city, api_key, session)
                                    )
                                    tasks.append(task)
                                
                                results = await asyncio.gather(*tasks)
                                return results
                        
                        with st.spinner("Получаем погоду для всех городов..."):
                            results = asyncio.run(fetch_all_cities())
                            
                            for result in results:
                                if 'error' not in result:
                                    st.write(f"**{result['city']}**: {result['temp']:.1f}°C - {result['description']}")
                                else:
                                    st.write(f"**{result['city']}**: Ошибка - {result['error']}")
            
            with col2:
                if season_stats is not None:
                    st.subheader("Исторические справочные данные")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Box(
                        y=city_data[city_data['season'] == current_season]['temperature'],
                        name=f'Историческое распределение для {current_season}',
                        boxpoints=False,
                        marker_color='lightblue'
                    ))
                    
                    fig.add_shape(
                        type="rect",
                        xref="paper",
                        yref="y",
                        x0=0, x1=1,
                        y0=season_stats['season_mean'] - sigma_threshold * season_stats['season_std'],
                        y1=season_stats['season_mean'] + sigma_threshold * season_stats['season_std'],
                        fillcolor="green",
                        opacity=0.2,
                        line_width=0
                    )
                    
                    fig.update_layout(
                        title=f'Распределение температур для {current_season} и нормальный диапазон',
                        yaxis_title='Температура (°C)',
                        height=300
                    )
                    
                    st.plotly_chart(fig, width='stretch')
        else:
            st.warning("Введите API-ключ OpenWeatherMap для получения данных о текущей погоде")
        
        st.divider()
        st.header("⚡ Анализ производительности")
        
        if st.checkbox("Показать сравнение производительности"):
            @timing_decorator
            def process_sequential(df_size):
                return load_and_process_data(st.session_state.uploaded_df, selected_city, window_size, False, True, df_size)
            
            @timing_decorator
            def process_parallel(df_size):
                return load_and_process_data(st.session_state.uploaded_df, selected_city, window_size, True, True, df_size)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Производительность обработки данных")
                st.info("Polars автоматически использует многопоточность для большинства операций")

                df_size = st.slider("Увеличьте в несколько раз размер данных для наилучшего результата теста", 1, 1000, 100)
                if st.button("Запустить тест производительности"):
                    with st.spinner("Тестируем последовательную обработку..."):
                        _, seq_time = process_sequential(df_size)
                    
                    with st.spinner("Тестируем параллельную обработку..."):
                        _, par_time = process_parallel(df_size)
                    
                    speedup = seq_time / par_time if par_time > 0 else 1

                    metrics_col1, metrics_col2 = st.columns(2)
                    with metrics_col1:
                        st.metric("Время последовательной обработки", f"{seq_time:.5f} сек")
                    with metrics_col2:
                        st.metric("Время параллельной обработки", f"{par_time:.5f} сек", 
                                 delta=f"Ускорение {seq_time-par_time:.5f} сек")
                    
                    if speedup > 1:
                        st.success(f"Polars ускорил выполнение в {speedup:.3f} раз")
                    else:
                        st.warning("Для малых объемов данных Polars может быть медленнее из-за накладных расходов")
            
            with col2:
                st.subheader("Производительность API-запросов")
                st.info(
                    "**Синхронные запросы**: Подходят для запросов к одному городу, просты в реализации\n\n"
                    "**Асинхронные запросы**: Эффективны для одновременных запросов к нескольким городам"
                )
                
                if api_key and st.button("Тест скорости API-запросов"):
                    sync_times = []
                    for _ in range(len(cities)):
                        start = time.time()
                        get_current_weather_sync(selected_city, api_key)
                        sync_times.append(time.time() - start)
                    
                    async def test_async_speed():
                        async with aiohttp.ClientSession() as session:
                            start = time.time()
                            tasks = [get_current_weather_async(city, api_key, session) for city in cities]
                            await asyncio.gather(*tasks)
                            return time.time() - start
                    
                    async_time = asyncio.run(test_async_speed())
                    
                    st.write(f"Время синхронного запроса для всех городов: {sum(sync_times):.2f} сек/город")
                    st.write(f"Среднее время синхронного запроса: {np.mean(sync_times):.2f} сек/город")
                    st.write(f"Время асинхронного запроса для всех городов: {async_time:.2f} сек")
                    st.write(f"Среднее время на город: {async_time / len(cities):.2f} сек")

if __name__ == "__main__":
    main()