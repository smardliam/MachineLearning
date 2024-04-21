import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
dt=np.dtype("f8,f8,f8,f8,U30")
data2=np.genfromtxt("iris.data",delimiter=",",dtype=dt)
# Загрузка данных из файла iris.data
sepal_length = [] # Длина чашелистника
sepal_width = [] # Ширина чашелистника
petal_length = [] # Длина лепестка
petal_width = [] # Ширина лепестка
# Проход по каждой строке данных data2
for dot in data2:
    sepal_length.append(dot[0])
    sepal_width.append(dot[1])
    petal_length.append(dot[2])
    petal_width.append(dot[3])

# Построение графика для сравнения ширины и длины чашелистника
plt.figure(1)

setosa, = plt.plot(sepal_length[:50],sepal_width[:50],'ro',label='Setosa')
versicolor, = plt.plot(sepal_length[50:100], sepal_width[50:100],'g^', label='Versicolor')
virginica, = plt.plot(sepal_length[100:150], sepal_width[100:150], 'bs', label='Verginica')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.xlabel('Длина чашелистника')
plt.ylabel("Ширина чашелистника")

# Построение графика для сравнения ширины и длины лепестка
plt.figure(2)
Setosa, = plt.plot(sepal_length[:50], petal_length[:50], 'ro', label='Setosa')

versicolor, = plt.plot(sepal_length[50:100], petal_length[50:100], 'g^', label='versicolor')
virginica, = plt.plot(sepal_length[100:150], petal_length[100:150], 'bs', label='Verginica')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.xlabel("Длина чашелистника")
plt.ylabel( 'Длина лепестка')

# Построение графика для сравнения ширины чашелистника и ширины лепестка
plt.figure(3)
setosa, = plt.plot(sepal_length[:50], petal_width[:50], 'ro', label='setosa')
versicolor, = plt.plot(sepal_length[50:100], petal_width[50:100], 'g^', label='Versicolor')
virginica, = plt.plot(sepal_length[100:150], petal_width[100:150], 'bs', label='Verginica')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel("Длина чашелистника")
plt.ylabel('Ширина лепестка')

# Построение графика для сравнения длины и ширины лепестка
plt.figure(4)
setosa, = plt.plot(petal_length[:50], petal_width[:50], 'ro', label='setosa')
versicolor, = plt.plot(petal_length[50:100], petal_width[50:100], 'g^', label='Versicolor')
virginica, = plt.plot(petal_length[100:150], petal_width[100:150], 'bs', label='Verginica')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel("Длина лепестка")
plt.ylabel('Ширина лепестка')
plt.show()
