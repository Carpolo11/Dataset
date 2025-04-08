import pandas as pd
import matplotlib.pyplot as plt # Importamos matplotlib para gráficos
import seaborn as sns           # Importamos seaborn para gráficos más estéticos

# --- Configuración de Estilo (Opcional, mejora la apariencia) ---
sns.set_theme(style="whitegrid") # Aplica un estilo general a los gráficos de Seaborn

# --- Carga del Dataset ---
# Asegúrate que la ruta sea correcta para tu sistema
try:
    # (Asegúrate de que esta ruta es correcta o ajústala si es necesario)
    df = pd.read_csv(r'C:\Users\linac\OneDrive\Documentos\datasett\archive\heart.csv')
except FileNotFoundError:
    print("Error: Archivo no encontrado en la ruta especificada.")
    print(r'Ruta intentada: C:\Users\linac\OneDrive\Documentos\datasett\archive\heart.csv')
    print("Por favor, verifica que el archivo exista en esa ubicación.")
    exit()
except Exception as e:
    print(f"Ocurrió un error al leer el archivo: {e}")
    exit()

# --- Análisis Exploratorio de Datos (EDA) ---

# 1. Descripción inicial
print("--- 1. Descripción Inicial ---")
print("Primeras filas del Dataset:")
print(df.head())
print("\nInformación General del Dataset:")
print(df.info())
print("\nEstadísticas Descriptivas (Variables Numéricas):")
print(df.describe())
print("-" * 50) # Separador

# 2. Visualización de distribuciones
print("\n--- 2. Visualizando Distribuciones Individuales ---")

# Histograma de la edad ('age')
plt.figure(figsize=(8, 5))
sns.histplot(df['age'], kde=True, bins=15) # Aumentar bins puede dar más detalle
plt.title('Distribución de la Edad')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

# Gráfico de barras para el sexo ('sex')
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='sex', data=df, palette='viridis') # Usar paleta de colores
plt.title('Distribución por Sexo')
plt.xlabel('Sexo')
plt.ylabel('Cantidad')
try:
    sex_labels = {0: 'Femenino', 1: 'Masculino'}
    ax.set_xticklabels([sex_labels.get(int(item.get_text()), item.get_text()) for item in ax.get_xticklabels()])
except Exception as e:
    print(f"No se pudieron aplicar etiquetas descriptivas al gráfico de sexo: {e}")
    plt.xlabel('Sexo (Valores originales)')
plt.tight_layout()
plt.show()

# (Considera añadir más gráficos aquí para otras variables importantes: chol, trestbps, thalach, cp, etc.)
# Ejemplo: Distribución del Colesterol
plt.figure(figsize=(8, 5))
sns.histplot(df['chol'], kde=True, color='orange')
plt.title('Distribución del Colesterol (chol)')
plt.xlabel('Colesterol (mg/dl)')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

print("-" * 50) # Separador

# 3. Identificación de correlaciones
print("\n--- 3. Matriz de Correlación ---")
try:
    correlation_matrix = df.corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Matriz de Correlación de Características')
    plt.tight_layout()
    plt.show()
except TypeError as e:
    print(f"\nError al calcular la matriz de correlación: {e}. Asegúrate que las columnas sean numéricas.")
except Exception as e:
    print(f"\nOcurrió un error inesperado al generar el heatmap: {e}")

print("-" * 50) # Separador

# 4. Análisis de relaciones entre características y la variable objetivo ('target')
print("\n--- 4. Análisis de Relaciones con la Variable Objetivo ('target') ---")
# (Asumimos que 'target' = 1 significa presencia de enfermedad, 0 ausencia. Verifica tu dataset.)

# Relación Edad vs Target
plt.figure(figsize=(8, 6))
sns.boxplot(x='target', y='age', data=df, palette='vlag') # Boxplot para comparar distribuciones
plt.title('Distribución de Edad por Diagnóstico (Target)')
plt.xlabel('Target (0: Sin Enfermedad, 1: Con Enfermedad)') # Ajusta etiquetas según tu data
plt.ylabel('Edad')
plt.tight_layout()
plt.show()

# Relación Sexo vs Target
plt.figure(figsize=(7, 5))
ax = sns.countplot(x='sex', hue='target', data=df, palette='pastel')
plt.title('Diagnóstico (Target) por Sexo')
plt.xlabel('Sexo')
plt.ylabel('Cantidad')
# Aplicar etiquetas descriptivas para Sexo
try:
    sex_labels = {0: 'Femenino', 1: 'Masculino'}
    ax.set_xticklabels([sex_labels.get(int(item.get_text()), item.get_text()) for item in ax.get_xticklabels()])
except Exception as e:
    print(f"No se pudieron aplicar etiquetas descriptivas al gráfico de sexo vs target: {e}")
    plt.xlabel('Sexo (Valores originales)')
# Añadir leyenda descriptiva para Target
handles, labels = ax.get_legend_handles_labels()
# Asume 0=No, 1=Sí. Ajusta si es diferente.
target_labels = { '0': 'Sin Enfermedad', '1': 'Con Enfermedad'}
ax.legend(handles=handles, title='Diagnóstico', labels=[target_labels.get(lbl, lbl) for lbl in labels])
plt.tight_layout()
plt.show()

# Relación Tipo de Dolor en el Pecho (cp) vs Target
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='cp', hue='target', data=df, palette='rocket')
plt.title('Diagnóstico (Target) por Tipo de Dolor en el Pecho (cp)')
plt.xlabel('Tipo de Dolor en el Pecho (cp)') # Considera añadir etiquetas descriptivas si conoces los tipos
plt.ylabel('Cantidad')
# Añadir leyenda descriptiva para Target
handles, labels = ax.get_legend_handles_labels()
target_labels = { '0': 'Sin Enfermedad', '1': 'Con Enfermedad'}
ax.legend(handles=handles, title='Diagnóstico', labels=[target_labels.get(lbl, lbl) for lbl in labels])
plt.tight_layout()
plt.show()


# Relación Colesterol (chol) vs Target
plt.figure(figsize=(8, 6))
sns.boxplot(x='target', y='chol', data=df, palette='coolwarm')
plt.title('Distribución de Colesterol por Diagnóstico (Target)')
plt.xlabel('Target (0: Sin Enfermedad, 1: Con Enfermedad)')
plt.ylabel('Colesterol (mg/dl)')
plt.tight_layout()
plt.show()

# Relación Máxima Frecuencia Cardíaca (thalach) vs Target
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='thalach', hue='target', kde=True, palette='muted') # Histograma con separación por target
plt.title('Distribución de Máx. Frecuencia Cardíaca por Diagnóstico (Target)')
plt.xlabel('Máxima Frecuencia Cardíaca Alcanzada (thalach)')
plt.ylabel('Frecuencia')
# Mejorar leyenda
handles, labels = plt.gca().get_legend_handles_labels()
target_labels = { '0': 'Sin Enfermedad', '1': 'Con Enfermedad'}
plt.legend(handles=handles, title='Diagnóstico', labels=[target_labels.get(lbl, lbl) for lbl in labels])
plt.tight_layout()
plt.show()

# --- Fin del Análisis Exploratorio ---
print("\n" + "="*60)
print("--- Fin del Análisis Exploratorio de Datos (EDA) ---")
print("="*60)

# --- 3. Preprocesamiento de Datos ---
print("\n" + "="*60)
print("--- 3. Preprocesamiento de Datos ---")
print("="*60)

# Este es el punto 3 de la 1ra parte: Imputación de valores faltantes (con la media)

# Verificar si hay valores faltantes en cada columna antes de la imputación
print("\nValores faltantes por columna antes de la imputación:")
print(df.isnull().sum())

# Imputar valores faltantes en columnas numéricas con la media
for column in df.select_dtypes(include=['number']).columns:
    if df[column].isnull().any():
        mean_value = df[column].mean()
        df[column].fillna(mean_value, inplace=True)
        print(f"Valores faltantes en '{column}' imputados con la media ({mean_value:.2f}).")
    else:
        print(f"La columna '{column}' no tiene valores faltantes.")

# Verificar si quedan valores faltantes después de la imputación
print("\nValores faltantes por columna después de la imputación:")
print(df.isnull().sum())

print("\n--- Fin del Preprocesamiento de Datos ---")
print("="*60)


