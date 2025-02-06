import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression  # Линейная регрессия
from sklearn.tree import DecisionTreeRegressor     # Дерево решений
from xgboost import XGBRegressor                  # Градиентный бустинг
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


# Функция для вычисления метрик
def evaluate_model(y_true, y_pred):
    """
    Вычисляет R2 и RMSE для предсказаний модели.
    :param y_true: истинные значения
    :param y_pred: предсказанные значения
    :return: R2 и RMSE
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, rmse

def regression_accuracy(y_true, y_pred, tolerance=0.1):
    """
    Вычисляет точность для регрессии как долю предсказаний,
    которые находятся в пределах допустимой погрешности.
    :param y_true: истинные значения
    :param y_pred: предсказанные значения
    :param tolerance: допустимая относительная ошибка (например, 0.1 = 10%)
    :return: точность (accuracy)
    """
    errors = np.abs((y_pred - y_true) / y_true)
    accuracy = np.mean(errors <= tolerance)
    return accuracy

def normalize_dataframe(df, columns=None):
    """
    Нормализует значения в указанных столбцах DataFrame в диапазон [0, 1].
    
    :param df: pandas DataFrame
    :param columns: Список столбцов для нормализации. Если None, нормализуются все числовые столбцы.
    :return: DataFrame с нормализованными значениями
    """
    df_normalized = df.copy()  # Создаем копию DataFrame, чтобы не изменять исходный
    if columns is None:
        # Если список столбцов не указан, выбираем все числовые столбцы
        columns = df.select_dtypes(include=['number']).columns
    
    for col in columns:
        df_normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    
    return df_normalized