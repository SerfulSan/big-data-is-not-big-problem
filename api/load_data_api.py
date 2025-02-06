import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import subprocess
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import os

def feature_importance_random_forest(df, target_column):
    """
    Функция для оценки важности признаков с помощью Random Forest.

    Параметры:
    df (pd.DataFrame): Входной DataFrame, содержащий данные.
    target_column (str): Название столбца с целевой переменной.

    Возвращает:
    feature_importances (pd.Series): Серия с важностью признаков.
    """
    # Разделение данных на признаки и целевую переменную
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Создание и обучение модели Random Forest
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)

    # Получение важности признаков
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    # Визуализация важности признаков
    plt.figure(figsize=(10, 6))
    feature_importances.plot(kind='bar', color='skyblue')
    plt.title('Feature Importances using Random Forest')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.show()

    return feature_importances

def optimize_dataframe(df):
    """
    Функция для оптимизации типов данных в DataFrame.
    
    Параметры:
        df (pd.DataFrame): Исходный DataFrame.
    
    Возвращает:
        pd.DataFrame: DataFrame с оптимизированными типами данных.
    """
    # Копируем исходный DataFrame
    optimized_df = df.copy()
    
    # Определяем начальный размер DataFrame
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2  # Размер в мегабайтах
    
    print(f"Начальный размер DataFrame: {initial_memory:.2f} MB")
    
    for col in optimized_df.columns:
        col_type = optimized_df[col].dtype
        
        # Если столбец числовой (int или float)
        if np.issubdtype(col_type, np.number):
            # Для целочисленных типов
            if np.issubdtype(col_type, np.integer):
                min_val = optimized_df[col].min()
                max_val = optimized_df[col].max()
                
                if min_val >= 0:
                    if max_val <= np.iinfo(np.uint8).max:
                        optimized_df[col] = optimized_df[col].astype(np.uint8)
                    elif max_val <= np.iinfo(np.uint16).max:
                        optimized_df[col] = optimized_df[col].astype(np.uint16)
                    elif max_val <= np.iinfo(np.uint32).max:
                        optimized_df[col] = optimized_df[col].astype(np.uint32)
                    else:
                        optimized_df[col] = optimized_df[col].astype(np.uint64)
                else:
                    if min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
                        optimized_df[col] = optimized_df[col].astype(np.int8)
                    elif min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
                        optimized_df[col] = optimized_df[col].astype(np.int16)
                    elif min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
                        optimized_df[col] = optimized_df[col].astype(np.int32)
                    else:
                        optimized_df[col] = optimized_df[col].astype(np.int64)
            
            # Для вещественных типов
            elif np.issubdtype(col_type, np.floating):
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        
        # Если столбец строковый
        elif col_type == 'object':
            num_unique_values = len(optimized_df[col].unique())
            num_total_values = len(optimized_df[col])
            
            # Если уникальных значений мало, преобразуем в категорию
            if num_unique_values / num_total_values < 0.5:
                optimized_df[col] = optimized_df[col].astype('category')
    
    # Определяем конечный размер DataFrame
    final_memory = optimized_df.memory_usage(deep=True).sum() / 1024**2  # Размер в мегабайтах
    reduction = (initial_memory - final_memory) / initial_memory * 100  # Процент уменьшения
    
    print(f"Конечный размер DataFrame: {final_memory:.2f} MB")
    print(f"Размер уменьшился на: {reduction:.2f}%")
    
    return optimized_df

def load_csv(file_path, num_rows=0):
    """
    Загружает данные из CSV-файла в DataFrame.

    Параметры:
        file_path (str): Путь к CSV-файлу.
        num_rows (int): Количество строк для загрузки. Если 0, загружается весь файл.

    Возвращает:
        pd.DataFrame: DataFrame с данными из файла.
    """
    try:
        # Если num_rows == 0, загружаем весь файл
        if num_rows == 0:
            df = pd.read_csv(file_path)
        else:
            # Иначе загружаем указанное количество строк
            df = pd.read_csv(file_path, nrows=num_rows)
        
        return df
    except Exception as e:
        print(f"Произошла ошибка при чтении файла: {e}")
        return None
    
def plot_distributions(df, bins=30, color='blue'):
    """
    Функция для построения графиков распределений для всех числовых столбцов в DataFrame.

    Параметры:
    df (pd.DataFrame): Входной DataFrame, содержащий данные.
    bins (int): Количество интервалов (бинов) для гистограммы (по умолчанию 30).
    color (str): Цвет графика (по умолчанию 'blue').
    """
    # Установка стиля seaborn
    sns.set(style="whitegrid")

    # Определение числовых столбцов
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    # Создание графиков распределений для каждого числового столбца
    for column in numeric_columns:
        plt.figure(figsize=(8, 4))  # Размер графика
        sns.histplot(df[column], kde=True, bins=bins, color=color)  # Гистограмма с KDE
        plt.title(f'Distribution of {column}')  # Заголовок графика
        plt.xlabel(column)  # Подпись оси X
        plt.ylabel('Frequency')  # Подпись оси Y
        plt.show()  # Отображение графика