import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset desde un archivo CSV
df = pd.read_csv(r'C:\Users\Ca908\Downloads\archive\heart.csv')

# Mostrar las primeras filas del dataset
print(df.head())

#Informacion general del dataset 
# Dimensiones del dataset
print(f"\nDimensiones del dataset: {df.shape}")

# Tipos de datos y valores nulos
print("\nInformación del dataset:")
print(df.info())

# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(df.describe())

#verificacion de valores nulos
print("\nValores nulos por columna:")
print(df.isnull().sum())

#Distribucion de variables
# Histograma de cada característica
df.hist(figsize=(15, 10), bins=20, color='skyblue', edgecolor='black')
plt.tight_layout()
plt.suptitle("Distribuciones de las Características", y=1.02)
plt.show()

#Matriz de correlacion 
# Heatmap de correlaciones
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Matriz de Correlación entre Características")
plt.show()

#Relacion entre variables y variable objetivo target
# Conteo de clases
sns.countplot(data=df, x='target', palette='Set2')
plt.title('Distribución de Clases (0 = Sano, 1 = Enfermo)')
plt.xlabel("Clase")
plt.ylabel("Cantidad")
plt.show()

# Boxplot entre edad y presencia de enfermedad
sns.boxplot(data=df, x='target', y='age', palette='Set3')
plt.title('Distribución de Edad por Clase')
plt.xlabel("Enfermedad (0 = No, 1 = Sí)")
plt.ylabel("Edad")
plt.show()

