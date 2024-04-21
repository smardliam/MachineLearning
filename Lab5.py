import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

dataset = pd.read_csv('Data5.csv')
dataset.head()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print("Матрица признаков:")
print(X)
print("Зависимая переменная:")
print(y)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

print("Матрица признаков без пропущенных значений:")
print(X)
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print("Зависимая переменная после кодирования:")
print(y)
X_dirty = X.copy()
print("Копия 'грязного' объекта:")
print(X_dirty)
transformers = [
    ('onehot', OneHotEncoder(), [0]),
    ('imp', SimpleImputer(), [1, 2])
]

ct = ColumnTransformer(transformers)

X_transformed = ct.fit_transform(X_dirty)
print("Размер преобразованных данных:")
print(X_transformed.shape)
print("Преобразованные данные:")
print(X_transformed)

X_data = pd.DataFrame(
    X_transformed,
    columns=['C1', 'C2', 'C3', 'Age', 'Salary']
)
print("Преобразованные данные в DataFrame:")
print(X_data)