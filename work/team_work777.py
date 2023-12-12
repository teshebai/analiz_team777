#%%
import pandas as pd
import numpy as np
import psycopg2
import seaborn as sns                       
import matplotlib.pyplot as plt            
import matplotlib_inline  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Открываем CSV файл
file_path = r'C:\Users\Nurda\Desktop\team_work\work\Global_Education.csv'

# CSV файл делаем датафрейм
df = pd.read_csv(r'C:\Users\Nurda\Desktop\team_work\work\Global_Education.csv', encoding='latin1')

#%%
print(df.head()) # экранга шыгарамыз датафрейм ретынде


#%% Посмотрим на основную информацию о датасете, такую как типы данных и наличие пропущенных значений
print(df.info())

#%% Посмотрим на основные статистические характеристики числовых данных
print(df.describe())

#%% Проверим наличие пропущенных значений
print(df.isnull().sum())

#%% Удаление дубликатов (если они есть)
df = df.drop_duplicates()

# Обработка пропущенных значений (например, заполнение их средними значениями или удаление)
#%% Например, для заполнения пропущенных значений в числовых столбцах средними значениями:
df.interpolate(inplace=True)  # Интерполяция для заполнения пропущенных значений

# Дополнительные шаги анализа и очистки могут включать в себя обработку категориальных данных, создание новых признаков и т. д.

#%% Сохранение очищенного датасета (если необходимо)
df.to_csv('new_dataset.csv', index=False)

######################################################
## 2) Провести детальный анализ (EDA)
#%%
print(df)
# %% 
df.head(5) #Чтобы отобразить 5 верхних строк
# %% 
df.tail(5) #Чтобы отобразить нижние 5 строк
# %% Проверка типов данных
df.dtypes


# %% 3) Сделать визулизацию важных переменных или деталей в датасете
df.hist(figsize=(15, 10)) #figsize задает размер фигуры (ширина, высота) .
plt.show() #Этот код отображает график, созданный методом hist


# %% Исследование категориальных переменных
sns.countplot(x='OOSR_Primary_Age_Female', data=df) #для создания столбчатой диаграммы.
plt.show()  #Этот код отображает график

# %% Сопоставьте различные объекты друг с другом (рассеяние), с частотой (гистограмма)
df.OOSR_Primary_Age_Female.value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title("OOSR_Primary_Age_Female by Countries and areas")
plt.ylabel('OOSR_Primary_Age_Female')  #y осьынде OOSR_Primary_Age_Female 
plt.xlabel('Countries and areas')  #x осьынде Countries and areas
# %% Heat Maps
# Исключаем нечисловые столбцы
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
numeric_df = df[numeric_columns]
# Создание тепловой карты
plt.figure(figsize=(15, 8))
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# %% Scatterplot
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(df['OOSR_Pre0Primary_Age_Male'], df['OOSR_Primary_Age_Female'])
ax.set_xlabel('OOSR_Pre0Primary_Age_Male')
ax.set_ylabel('OOSR_Primary_Age_Female')
plt.show()

#####################################################
# %% 
# Визуализация отношения
plt.scatter(df['OOSR_Pre0Primary_Age_Male'], df['Completion_Rate_Primary_Male'])
plt.xlabel('OOSR_Pre0Primary_Age_Male')
plt.ylabel('Completion_Rate_Primary_Male')
plt.title('Связь между OOSR_Primary_Age_Female и Completion_Rate_Primary_Male')
plt.show()

# %% Тестирование гипотезы
from scipy.stats import pearsonr

# Нулевая гипотеза: Нет корреляции
# Альтернативная гипотеза: Есть положительная корреляция
stat, p_value = pearsonr(df['OOSR_Primary_Age_Female'], df['Completion_Rate_Primary_Male'])

print(f'Коэффициент корреляции: {stat}\nЗначение p: {p_value}')

# Проверьте, если ли p-value ниже уровня значимости (например, 0,05)
if p_value < 0.05:
    print("Отклонить нулевую гипотезу")
else:
    print("Не удалось отклонить нулевую гипотезу")



# %% Построения регрессии
#################################################
# Загрузите ваш датасет
file_path = r'C:\Users\Nurda\Desktop\team_work\work\Global_Education.csv'

df = pd.read_csv(r'C:\Users\Nurda\Desktop\team_work\work\Global_Education.csv', encoding='latin1')

# Исключаем нечисловые столбцы
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
numeric_df = df[numeric_columns]

# Выбираем целевую переменную и признаки
target_column = 'Longitude' 
features = numeric_df.drop(target_column, axis=1)
target = numeric_df[target_column]

# Разделение на тренировочный и тестовый наборы данных
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Создание модели линейной регрессии
model = LinearRegression()

# Обучение модели на тренировочных данных
model.fit(X_train, y_train)

# Предсказание на тестовых данных
y_pred = model.predict(X_test)

# Визуализация результатов
plt.scatter(y_test, y_pred, color='black')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='blue', linewidth=3)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.show()

# Оценка качества модели
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# %%
