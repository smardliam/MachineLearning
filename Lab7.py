import numpy as np
import pandas as pd

# Загрузка данных
данные = pd.read_csv('50_Startups.csv')

# Применение One-Hot Encoding для переменной 'State'
данные = pd.get_dummies(данные, columns=['State'])

# Преобразование логических значений в целые числа
for колонка in данные.columns[-3:]:
    данные[колонка] = данные[колонка].astype(int)

# Вывод первых строк данных для проверки
print(данные.head())

# Разделение на признаки и целевую переменную
X = данные.drop(columns=['Profit']).values
y = данные['Profit'].values

# Разделение на обучающий и тестовый наборы
from sklearn.model_selection import train_test_split
X_обуч, X_тест, y_обуч, y_тест = train_test_split(X, y, test_size=0.2, random_state=0)

# Обучение модели линейной регрессии
from sklearn.linear_model import LinearRegression
регрессор = LinearRegression()
регрессор.fit(X_обуч, y_обуч)

# Предсказание результатов на тестовом наборе
y_пред = регрессор.predict(X_тест)

# Вывод предсказанных результатов
print(y_пред)

# Оценка модели
import statsmodels.api as sm
X = sm.add_constant(X)
модель = sm.OLS(y, X).fit()
print(модель.summary())

# Отбор признаков
X_опт = X[:, [0, 1, 3, 4, 5]]
регрессор_OLS = sm.OLS(endog=y, exog=X_опт).fit()
print(регрессор_OLS.summary())

X_опт = X[:, [0, 3, 4, 5]]
регрессор_OLS = sm.OLS(endog=y, exog=X_опт).fit()
print(регрессор_OLS.summary())

X_опт = X[:, [0, 3, 5]]
регрессор_OLS = sm.OLS(endog=y, exog=X_опт).fit()
print(регрессор_OLS.summary())

X_опт = X[:, [0, 3]]
регрессор_OLS = sm.OLS(endog=y, exog=X_опт).fit()
print(регрессор_OLS.summary())
