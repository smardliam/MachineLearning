import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


data_path = "telecom_churn.csv"
data = pd.read_csv(data_path)
data.info();
plt.figure(1)
plt.bar(data.index,data['Total day minutes']);
plt.figure(2)
hist=data['Total day minutes'].value_counts()
plt.bar(hist.index,hist);
plt.figure(3)
sns.boxplot(data['Total day minutes'])
plt.figure(4)
top_data = data[['State','Total day minutes']]
top_data = top_data.groupby('State').sum()
top_data = top_data.sort_values('Total day minutes',ascending=False)
top_data = top_data[:3].index.values
sns.boxplot(y='State',
            x='Total day minutes',
            data=data[data.State.isin(top_data)], palette='Set3');
feats = [f for f in data.columns if 'charge' in f]
print(feats)
sns.pairplot(data[feats])
sns.pairplot(data[feats+['Churn']],hue='Churn')
plt.figure()
plt.scatter(data['Total day charge'],
            data['Total intl charge'],
            color='lightblue', edgecolors='blue')
plt.xlabel('Дневные начисления')
plt.ylabel('Международные начисления')
plt.title('Распределение по двум признакам');
plt.figure()
# Раскрашивание данных
# Цвет в зависимости от ухода клиента
# Раскраска лояльных и ушедших клиентов,
# добавление легенды

# Ушедшие клиенты
data_churn = data[data['Churn']]
# Оставшиеся клиенты
data_loyal = data[~data['Churn']]

plt.scatter(data_churn['Total day charge'],
            data_churn['Total intl charge'],
            color='orange',
            edgecolors='red',
            label='Ушли'
           )
plt.scatter(data_loyal['Total day charge'],
            data_loyal['Total intl charge'],
            color='lightblue',
            edgecolors='blue',
            label='Остались'
           )
plt.xlabel('Дневные начисления')
plt.ylabel('Международные начисления')
plt.title('Распределение клиентов')
plt.legend()


plt.show()
sns.heatmap(data.corr(),cmap=plt.cm.Blues)
