import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

data_source = 'iris.data'
# Загрузка данных из файла iris.data
d = pd.read_table(data_source, delimiter=',',
                  header=None,
                  names=['sepal_length','sepal_width',
                         'petal_length','petal_width','answer'])
print(d.head())  # Вывод первых строк таблицы
d.info()
# Визуализация взаимосвязи между признаками с разделением по классам
sns.pairplot(d, hue='answer', markers=["o", "s", "D"])

# Обучающие данные
X_train = d[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_train = d['answer']

K = 3

# Создание и настройка классификатора
knn = KNeighborsClassifier(n_neighbors=K)
# Обучение модели классификатора
knn.fit(X_train.values, y_train)

# Использование классификатора
# Определение признаков для нового объекта
X_test = np.array([[1.2, 1.0, 2.8, 1.2]])
# Получение прогноза для нового объекта
target = knn.predict(X_test)

# Разделение данных на обучающий и тестовый наборы
X_train, X_holdout, y_train, y_holdout = train_test_split(
    d.iloc[:, 0:4],
    d['answer'],
    test_size=0.3,
    random_state=17)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_holdout)
accur = accuracy_score(y_holdout, knn_pred)
print('Точность: ', accur)
print(target)

# Значения параметра K
k_list = list(range(1, 50))
# Пустой список для хранения значений точности
cv_scores = []
# Подбор оптимального значения параметра K с помощью кросс-валидации
for K in k_list:
    knn = KNeighborsClassifier(n_neighbors=K)
    scores = cross_val_score(knn, d.iloc[:, 0:4], d['answer'], cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# Вычисление ошибки классификации
MSE = [1 - x for x in cv_scores]
plt.figure()
# Построение графика
plt.plot(k_list, MSE)
plt.xlabel('Количество соседей (K)')
plt.ylabel('Ошибка классификации (MSE)')

# Поиск оптимального значения параметра K
k_min = min(MSE)
all_k_min = []
for i in range(len(MSE)):
    if MSE[i] <= k_min:
        all_k_min.append(k_list[i])

# Вывод оптимальных значений K
print('Оптимальные значения K: ', all_k_min)
plt.show()
