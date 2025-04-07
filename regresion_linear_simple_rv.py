# Se recoge un dataset de Kaggle link: https://www.kaggle.com/datasets/monicahjones/steps-tracker-dataset
# Este conjunto de datos contiene el conteo diario de pasos, horas de sueño, minutos activos y calorías quemadas, 
# recopilados como parte del seguimiento personal de fitness

# Importar las bibliotecas necesarias
import numpy as np  # Para operaciones matemáticas y manejo de matrices
import matplotlib.pyplot as plt  # Para visualización de datos
import pandas as pd  # Para manipulación y análisis de datos

# Importar el dataset
# El Dataset 'IADataSet.xlsx'  debe estar en el mismo nivel de ruta o carpeta que este script(Mismo Directorio)
dataset = pd.read_excel('IADataSet.xlsx')
X = dataset.iloc[:, :-1].values  # Variable independiente (Pasos Recorridos)
y = dataset.iloc[:, -1].values  # Variable dependiente (Calorias quemadas)

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
from sklearn.model_selection import train_test_split
# Usamos 2/3 para entrenamiento y 1/3 para prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Entrenar el modelo de Regresión Lineal Simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  # Crear el modelo de regresión lineal
regressor.fit(X_train, y_train)  

# Predecir los resultados del conjunto de prueba
y_pred = regressor.predict(X_test) 

# Visualizar los resultados del conjunto de entrenamiento
plt.scatter(X_train, y_train, color='orange')  
plt.plot(X_train, regressor.predict(X_train), color='blue')  
plt.title('Pasos Recorridos vs Calorias Quemadas  (Conjunto de Entrenamiento)')  
plt.xlabel('Pasos Recorridos')  
plt.ylabel('Calorias Quemadas')  
plt.show()

# Visualizar los resultados del conjunto de prueba
plt.scatter(X_test, y_test, color='red')  
plt.plot(X_train, regressor.predict(X_train), color='blue')  
plt.title('Pasos Recorridos vs Calorias Quemadas (Conjunto de Prueba)')  
plt.xlabel('Pasos Recorridos')  
plt.ylabel('Calorias Quemadas')  
plt.show()  