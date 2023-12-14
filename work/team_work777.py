#%%
import pandas as pd
import numpy as np
import psycopg2
import seaborn as sns                       
import matplotlib.pyplot as plt            
import matplotlib_inline  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Открываем CSV файл
file_path = r'C:\Users\Nurda\Desktop\team_work\work\Global_Education.csv'

# CSV файл делаем датафрейм
df = pd.read_csv(r'C:\Users\Nurda\Desktop\team_work\work\Global_Education.csv', encoding='latin1')
#%%
print(df.head()) # экранга шыгарамыз датафрейм ретынде
df.shape
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

#%%
# Страны и районы: Название стран и районов.
# Широта: Широтные координаты географического местоположения.
# Долгота: Долготные координаты географического местоположения.
# OOSR_Pre0Primary_Age_Male: показатель не посещаемости школы мальчиками дошкольного возраста.
# OOSR_Pre0Primary_Age_Female: Показатель не посещаемости школы для девочек дошкольного возраста.
# OOSR_Primary_Age_Male: показатель не посещаемости школы для мальчиков младшего возраста.
# OOSR_Primary_Age_Female: Показатель не посещаемости школы женщинами младшего возраста.
# OOSR_Lower_Secondary_Age_Male: показатель не посещаемости школы мужчинами младшего среднего возраста.
# OOSR_Lower_Secondary_Age_Female: Показатель не посещаемости школы женщинами младших классов средней школы.
# OOSR_Upper_Secondary_Age_Male: показатель не посещаемости школы мужчинами старших классов средней школы.
# OOSR_Upper_Secondary_Age_Female: доля женщин, не посещающих школу в старших классах средней школы.
# Показатель завершения_примечания_мале: показатель завершения начального образования среди мужчин.
# Completion_Rate_Primary_Female: Показатель завершения начального образования среди женщин.
# Completion_Rate_Lower_Secondary_Male: показатель завершения неполного среднего образования среди мужчин.
# завершение_рейтинг_лучшей_секундной_женщины: показатель завершения неполной средней школы среди женщин.
# завершение_рейтинг_лучшей_секундной_женщины: показатель завершения полной средней школы среди мужчин.
# Уровень завершения_среднего образования для женщин: уровень завершения высшего среднего образования среди женщин.
# Grade_2_3_Proficiency_Reading: Уровень владения чтением для учащихся 2-3 классов.
# Grade_2_3_Proficiency_Math: Уровень владения математикой для учащихся 2-3 классов.
# Начальный уровень владения чтением: Уровень владения чтением в конце начального образования.
# Primary_End_Proficiency_Math: Владение математикой в конце начального образования.
# Lower_Secondary_End_Proficiency_Reading: Владение чтением в конце неполного среднего образования.
# Lower_Secondary_End_Proficiency_Math: Уровень владения математикой по окончании неполной средней школы.
# Youth_15_24_Literacy_Rate_Male: уровень грамотности среди молодых мужчин в возрасте 15-24 лет.
# Youth_15_24_Literacy_Rate_Female: Уровень грамотности среди молодых женщин в возрасте 15-24 лет.
# Коэффициент рождаемости: коэффициент рождаемости в соответствующих странах/районах.
# Общее количество учащихся в системе начального образования: общее количество учащихся в системе начального образования.
# Общее количество поступающих в высшие учебные заведения.
# Уровень безработицы_: Уровень безработицы в соответствующих странах/регионах.

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


#%%
##диаграммы для отображения уровней образования по полу.
sns.barplot(x='Countries and areas', y='OOSR_Primary_Age_Male', data=df[:20])
plt.title('Out-of-School Rate for Primary Age Males')
plt.xlabel('Country/Area')
plt.ylabel('Out-of-School Rate')
plt.xticks(rotation=90)
plt.show()



# %%
sns.boxplot(x='Completion_Rate_Primary_Male', y='Unemployment_Rate', data=df)
plt.title('Box Plot: Completion Rate Primary Male vs Unemployment Rate')
plt.show()

# %%
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
stat, p_value = pearsonr(df['Primary_End_Proficiency_Math'], df['Unemployment_Rate'])

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
target_column = 'Unemployment_Rate' 
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
model2 = DecisionTreeRegressor(random_state=42)
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)

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
model3 = RandomForestRegressor(random_state=42)
model3.fit(X_train, y_train)
y_pred = model3.predict(X_test)

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
